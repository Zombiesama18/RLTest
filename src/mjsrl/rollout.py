"""Full-episode rollouts; transitions link a seat's successive decision points."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from .engine import RiichiEnvAdapter
from .features import collate
from .schema import MahjongState, Transition


class RandomAgent:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def act(self, state):
        return self.rng.randrange(len(state.legal_actions)), {}


class HeuristicAgent(RandomAgent):
    """Small shanten baseline for smoke data; it is not expert human play."""

    def act(self, state):
        from riichienv import calculate_shanten

        from .schema import Tile

        for i, action in enumerate(state.legal_actions):
            if action.type in ("ron", "tsumo", "riichi_discard"):
                return i, {}
        candidates = []
        for i, action in enumerate(state.legal_actions):
            if action.type == "pass":
                return i, {}
            if action.type != "discard":
                continue
            hand = list(state.hand)
            hand.remove(action.tile)
            counts, tiles = {}, []
            for t in hand:
                tid = Tile.parse(t).tile34
                offset = counts.get(tid, 0)
                tiles.append(tid * 4 + offset)
                counts[tid] = offset + 1
            candidates.append((calculate_shanten(tiles), self.rng.random(), i))
        return (min(candidates)[2], {}) if candidates else super().act(state)


class ModelAgent:
    def __init__(self, model, device="cpu", stochastic=True):
        self.model, self.device, self.stochastic = model, device, stochastic

    @torch.no_grad()
    def act(self, state: MahjongState):
        self.model.eval()
        output = self.model(collate([state], self.device, self.model.config.history_length))
        dist = torch.distributions.Categorical(logits=output["logits"][0])
        index = dist.sample() if self.stochastic else output["logits"][0].argmax()
        return int(index), {
            "log_prob": float(dist.log_prob(index)),
            "value": float(output["value"][0]),
        }


@dataclass
class GameResult:
    transitions: list[Transition]
    scores: tuple[int, ...]
    ranks: tuple[int, ...]
    events: list[dict]
    steps: int


def play_game(
    agents: dict,
    seed: int,
    mode="4p-red-half",
    gamma=0.997,
    reward_mode="combined",
    max_steps=20000,
) -> GameResult:
    env = RiichiEnvAdapter(mode)
    states = env.reset(seed)
    initial_scores = env.scores()
    pending, records, step = {}, [], 0
    while not env.done():
        if step >= max_steps:
            raise RuntimeError("Episode limit exceeded; refusing to mark a truncation terminal")
        selected = {}
        for seat, state in states.items():
            if seat in pending:
                old_state, action, old_step, metadata = pending.pop(seat)
                elapsed = max(1, step - old_step)
                records.append(
                    Transition(
                        env.game_id,
                        old_step,
                        seat,
                        old_state,
                        action,
                        state,
                        0.0,
                        gamma**elapsed,
                        False,
                        elapsed,
                        metadata,
                    )
                )
            index, metadata = agents[seat].act(state)
            if not 0 <= index < len(state.legal_actions):
                raise ValueError("Agent returned an illegal candidate")
            pending[seat] = state, index, step, metadata
            selected[seat] = state.legal_actions[index]
        states = env.step(selected)
        step += 1
    ranks, scores = env.ranks(), env.scores()
    for seat, (state, action, old_step, metadata) in pending.items():
        reward = terminal_reward(scores[seat] - initial_scores[seat], ranks[seat], reward_mode)
        elapsed = max(1, step - old_step)
        records.append(
            Transition(
                env.game_id,
                old_step,
                seat,
                state,
                action,
                None,
                reward,
                0.0,
                True,
                elapsed,
                metadata,
            )
        )
    records.sort(key=lambda t: (t.decision_id, t.seat))
    return GameResult(records, scores, ranks, env.native.mjai_log, step)


def terminal_reward(delta: int, rank: int, mode="combined") -> float:
    rank_rewards = {1: 1.0, 2: 0.25, 3: -0.25, 4: -1.0}
    if mode not in ("combined", "rank", "points"):
        raise ValueError("Unknown reward mode")
    points = max(-2, min(2, delta / 8000)) if mode != "rank" else 0
    return points + (rank_rewards[rank] if mode != "points" else 0)
