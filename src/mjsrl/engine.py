"""Pinned RiichiEnv adapter; it alone owns native legal actions and hidden state."""

from __future__ import annotations

import json
from typing import Protocol

from riichienv import GameRule, RiichiEnv

from .schema import Action, MahjongState, Tile

NATIVE_TYPES = {
    "DISCARD": "discard",
    "RIICHI": "riichi_discard",
    "CHI": "chi",
    "PON": "pon",
    "DAIMINKAN": "daiminkan",
    "ANKAN": "ankan",
    "KAKAN": "kakan",
    "RON": "ron",
    "TSUMO": "tsumo",
    "PASS": "pass",
    "KYUSHU_KYUHAI": "kyuushu_kyuuhai",
}


class MahjongEngine(Protocol):
    def reset(self, seed: int) -> dict[int, MahjongState]: ...
    def step(self, actions: dict[int, Action]) -> dict[int, MahjongState]: ...
    def done(self) -> bool: ...
    def scores(self) -> tuple[int, ...]: ...
    def ranks(self) -> tuple[int, ...]: ...


def public_events(events: list, seat: int) -> list[dict]:
    result = []
    for raw in events:
        event = json.loads(raw) if isinstance(raw, str) else raw
        clean = {
            k: event[k]
            for k in ("type", "actor", "pai", "consumed", "target", "tsumogiri", "dora_marker")
            if k in event
        }
        if event["type"] == "tsumo" and event.get("actor") != seat:
            clean["pai"] = "?"
        # Never retain tehais, ura markers, IDs, account fields or oracle hands.
        result.append(clean)
    return result


def canonical_action(native, obs, history: list[dict]) -> Action:
    kind = NATIVE_TYPES[str(native.action_type).split(".")[-1]]
    tile = str(Tile.from136(native.tile)) if native.tile is not None else None
    consumed = list(native.consume_tiles)
    # Replay iterator sometimes supplies the full meld, including its claimed tile.
    expected = {"chi": 2, "pon": 2, "daiminkan": 3}.get(kind)
    if expected and len(consumed) == expected + 1 and native.tile in consumed:
        consumed.remove(native.tile)
    consumed = tuple(sorted(str(Tile.from136(t)) for t in consumed))
    source = None
    if kind in ("chi", "pon", "daiminkan", "ron"):
        for event in reversed(history):
            if event.get("type") in ("dahai", "kakan", "ankan"):
                source = event["actor"]
                break
    tsumogiri = kind in ("discard", "riichi_discard") and native.tile == obs.drawn_tile
    identity = json.dumps([kind, tile, consumed, source, tsumogiri], separators=(",", ":"))
    return Action(identity, kind, tile, consumed, source, tsumogiri)


def snapshot(obs, history: list[dict], game_id: str, mode: str) -> tuple[MahjongState, dict]:
    seat = obs.player_id
    native_map = {}
    for native in obs.legal_actions():
        action = canonical_action(native, obs, history)
        native_map.setdefault(action.id, (action, native))
    rivers = [list(river) for river in obs.discards]
    melds = []
    for player_melds in obs.melds:
        result = []
        for meld in player_melds:
            kind = str(meld.meld_type).split(".")[-1].lower()
            result.append(
                {
                    "type": kind,
                    "tiles": [str(Tile.from136(t)) for t in meld.tiles],
                    "from_seat": meld.from_who,
                }
            )
            # Engine keeps called discards in the river. Count/display each physical tile once.
            if meld.called_tile is not None and meld.from_who is not None:
                river = rivers[meld.from_who]
                if meld.called_tile in river:
                    river.remove(meld.called_tile)
        melds.append(tuple(result))
    # Histories are per hand. Public draw count includes replacement draws.
    draws = sum(e.get("type") == "tsumo" for e in history)
    state = MahjongState(
        seat,
        tuple(str(Tile.from136(t)) for t in obs.hand),
        tuple(obs.scores),
        tuple(tuple(str(Tile.from136(t)) for t in river) for river in rivers),
        tuple(melds),
        tuple(obs.riichi_declared),
        tuple(str(Tile.from136(t)) for t in obs.dora_indicators),
        obs.oya,
        obs.round_wind,
        obs.kyoku_index % 4 + 1,
        obs.honba,
        obs.riichi_sticks,
        max(0, 70 - draws),
        tuple(x[0] for x in native_map.values()),
        tuple(history),
        game_id,
        mode,
    )
    state.validate()
    return state, {key: item[1] for key, item in native_map.items()}


def update_history(history: list[dict], obs) -> list[dict]:
    events = public_events(obs.new_events(), obs.player_id)
    for event in events:
        if event["type"] == "start_kyoku":
            history = []
        history.append(event)
    return history


class RiichiEnvAdapter:
    def __init__(self, mode: str = "4p-red-half"):
        if mode not in ("4p-red-single", "4p-red-east", "4p-red-half"):
            raise ValueError("Only four-player red-tile modes are supported")
        self.mode = mode
        self.native = RiichiEnv(game_mode=mode, rule=GameRule.default_mjsoul())
        self.histories = {seat: [] for seat in range(4)}
        self.action_maps = {}
        self.game_id = ""

    def _convert(self, observations):
        states, self.action_maps = {}, {}
        for seat, obs in observations.items():
            self.histories[seat] = update_history(self.histories[seat], obs)
            if not obs.legal_actions():
                continue
            states[seat], self.action_maps[seat] = snapshot(
                obs,
                self.histories[seat],
                self.game_id,
                self.mode,
            )
        return states

    def reset(self, seed: int) -> dict[int, MahjongState]:
        self.histories = {seat: [] for seat in range(4)}
        self.game_id = f"sim:{self.mode}:{seed}"
        # In 0.4.10 reset(seed=...) does not reseed a randomly constructed instance.
        # Construct with the seed so independent adapters receive the same wall.
        self.native = RiichiEnv(game_mode=self.mode, seed=seed, rule=GameRule.default_mjsoul())
        return self._convert(self.native.reset())

    def step(self, actions: dict[int, Action]) -> dict[int, MahjongState]:
        if set(actions) != set(self.action_maps):
            raise ValueError("Must select one action for each active player")
        selected = {}
        for seat, action in actions.items():
            if action.id not in self.action_maps[seat]:
                raise ValueError(f"Illegal action for player {seat}")
            selected[seat] = self.action_maps[seat][action.id]
        return self._convert(self.native.step(selected))

    def done(self):
        return self.native.done()

    def scores(self):
        return tuple(self.native.scores())

    def ranks(self):
        return tuple(self.native.ranks())
