"""Encode only the explicit public schema; never encode raw simulator dictionaries."""

from __future__ import annotations

import torch

from .schema import ACTION_TYPES, MahjongState, Tile

GLOBAL_DIM = 18
EVENT_TYPES = (
    "tsumo",
    "dahai",
    "chi",
    "pon",
    "daiminkan",
    "ankan",
    "kakan",
    "reach",
    "reach_accepted",
    "dora",
    "hora",
    "ryukyoku",
)


def encode_state(state: MahjongState, history_length: int = 64) -> tuple[list, list]:
    seat = state.perspective_seat
    relative = lambda player: (player - seat) % 4  # noqa: E731
    # Each token: kind, tile37+1 (0=unknown), relative seat+1, flag.
    tokens = [[1, Tile.parse(t).tile37 + 1, 1, 0] for t in state.hand]
    tokens += [[2, Tile.parse(t).tile37 + 1, 0, 0] for t in state.dora_indicators]
    for player in range(4):
        tokens += [
            [3, Tile.parse(t).tile37 + 1, relative(player) + 1, 0] for t in state.rivers[player]
        ]
        for meld in state.melds[player]:
            tokens += [
                [4, Tile.parse(t).tile37 + 1, relative(player) + 1, int(meld["type"] == "ankan")]
                for t in meld["tiles"]
            ]
    for event in state.history[-history_length:]:
        kind = event.get("type")
        if kind not in EVENT_TYPES:
            continue
        actor = event.get("actor")
        player = relative(actor) + 1 if actor is not None else 0
        tile = event.get("pai", event.get("dora_marker"))
        # An accidental full-information replay must still hide opponents' draws.
        if kind == "tsumo" and actor != seat:
            tile = None
        tid = Tile.parse(tile).tile37 + 1 if tile and tile != "?" else 0
        tokens.append(
            [5 + EVENT_TYPES.index(kind), tid, player, int(bool(event.get("tsumogiri", False)))]
        )
    if not tokens:
        tokens = [[1, 0, 0, 0]]
    order = [(seat + i) % 4 for i in range(4)]
    numeric = [state.scores[i] / 30000 for i in order]
    numeric += [float(state.riichi[i]) for i in order]
    numeric += [
        relative(state.dealer) / 3,
        state.round_wind / 3,
        state.kyoku / 4,
        state.honba / 10,
        state.riichi_sticks / 10,
        state.wall_remaining / 70,
    ]
    numeric += [len(state.melds[i]) / 4 for i in order]
    return tokens, numeric


def encode_actions(state: MahjongState) -> list[list[int]]:
    encoded = []
    for a in state.legal_actions:
        tile = Tile.parse(a.tile).tile37 + 1 if a.tile else 0
        consumed = [Tile.parse(t).tile37 + 1 for t in a.consume]
        if len(consumed) > 4:
            raise ValueError("At most four consumed tiles per action")
        source = (a.from_seat - state.perspective_seat) % 4 + 1 if a.from_seat is not None else 0
        encoded.append(
            [
                ACTION_TYPES.index(a.type),
                tile,
                *(consumed + [0] * (4 - len(consumed))),
                source,
                int(a.tsumogiri),
            ]
        )
    return encoded


def collate(states: list[MahjongState], device="cpu", history_length: int = 64) -> dict:
    if not states:
        raise ValueError("Cannot collate empty state batch")
    encodings = [encode_state(s, history_length) for s in states]
    length = max(len(x[0]) for x in encodings)
    candidates = max(len(s.legal_actions) for s in states)
    tokens = torch.zeros(len(states), length, 4, dtype=torch.long)
    padding = torch.ones(len(states), length, dtype=torch.bool)
    actions = torch.zeros(len(states), candidates, 8, dtype=torch.long)
    mask = torch.zeros(len(states), candidates, dtype=torch.bool)
    for i, (state, (tok, _)) in enumerate(zip(states, encodings, strict=True)):
        tokens[i, : len(tok)] = torch.tensor(tok)
        padding[i, : len(tok)] = False
        actions[i, : len(state.legal_actions)] = torch.tensor(encode_actions(state))
        mask[i, : len(state.legal_actions)] = True
    return {
        "tokens": tokens.to(device),
        "padding": padding.to(device),
        "globals": torch.tensor([x[1] for x in encodings], dtype=torch.float32, device=device),
        "actions": actions.to(device),
        "legal_mask": mask.to(device),
    }
