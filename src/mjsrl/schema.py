"""Versioned public observations, tiles and variable legal action candidates."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "mjstate/1.0"
RULESET = "majsoul_4p_standard"
ACTION_TYPES = (
    "discard",
    "riichi_discard",
    "chi",
    "pon",
    "daiminkan",
    "ankan",
    "kakan",
    "ron",
    "tsumo",
    "pass",
    "kyuushu_kyuuhai",
)
MJAI_TYPES = {
    "dahai": "discard",
    "reach": "riichi_discard",
    "chi": "chi",
    "pon": "pon",
    "daiminkan": "daiminkan",
    "ankan": "ankan",
    "kakan": "kakan",
    "hora": "ron",
    "none": "pass",
    "ryukyoku": "kyuushu_kyuuhai",
}
HONORS = {"E": "1z", "S": "2z", "W": "3z", "N": "4z", "P": "5z", "F": "6z", "C": "7z"}


@dataclass(frozen=True)
class Tile:
    tile34: int
    is_red: bool = False

    def __post_init__(self):
        if not 0 <= self.tile34 < 34:
            raise ValueError("tile34 must be in 0..33")
        if self.is_red and self.tile34 not in (4, 13, 22):
            raise ValueError("Only suited fives can be red")

    @classmethod
    def parse(cls, value: str | dict | Tile) -> Tile:
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        value = HONORS.get(value, value)
        if not re.fullmatch(r"[1-9][mps]|[1-7]z|0[mps]|5[mps]r", value):
            raise ValueError(f"Invalid tile: {value}")
        if len(value) not in (2, 3) or value[1] not in "mpsz":
            raise ValueError(f"Invalid tile: {value}")
        red = value[0] == "0" or value.endswith("r")
        rank = 5 if value[0] == "0" else int(value[0])
        suit = "mpsz".index(value[1])
        if not 1 <= rank <= (7 if suit == 3 else 9):
            raise ValueError(f"Invalid tile: {value}")
        if len(value) == 3 and value[2] != "r":
            raise ValueError(f"Invalid tile: {value}")
        return cls(suit * 9 + rank - 1, red)

    @classmethod
    def from136(cls, tid: int) -> Tile:
        if not 0 <= tid < 136:
            raise ValueError("tile136 must be in 0..135")
        return cls(tid // 4, tid in (16, 52, 88))

    @property
    def tile37(self) -> int:
        return 34 + (self.tile34 // 9) if self.is_red else self.tile34

    def __str__(self) -> str:
        return f"{self.tile34 % 9 + 1}{'mpsz'[self.tile34 // 9]}{'r' if self.is_red else ''}"


@dataclass(frozen=True)
class Action:
    id: str
    type: str
    tile: str | None = None
    consume: tuple[str, ...] = ()
    from_seat: int | None = None
    tsumogiri: bool = False

    def __post_init__(self):
        if self.type not in ACTION_TYPES:
            raise ValueError(f"Unsupported action type {self.type}")
        if self.tile is not None:
            Tile.parse(self.tile)
        for tile in self.consume:
            Tile.parse(tile)
        if self.from_seat is not None and not 0 <= self.from_seat < 4:
            raise ValueError("from_seat must be 0..3")

    @classmethod
    def from_mjai(cls, event: dict[str, Any]) -> Action:
        kind = MJAI_TYPES[event["type"]]
        if event["type"] == "hora":
            kind = "tsumo" if event.get("actor") == event.get("target") else "ron"
        tile = event.get("pai")
        tile = str(Tile.parse(tile)) if tile and tile != "?" else None
        consumed = tuple(sorted(str(Tile.parse(t)) for t in event.get("consumed", [])))
        source = event.get("target")
        tsumogiri = bool(event.get("tsumogiri", False))
        identity = json.dumps([kind, tile, consumed, source, tsumogiri], separators=(",", ":"))
        return cls(identity, kind, tile, consumed, source, tsumogiri)

    @classmethod
    def from_dict(cls, value: dict) -> Action:
        return cls(**{**value, "consume": tuple(value.get("consume", ()))})


@dataclass(frozen=True)
class MahjongState:
    perspective_seat: int
    hand: tuple[str, ...]
    scores: tuple[int, ...]
    rivers: tuple[tuple[str, ...], ...]
    melds: tuple[tuple[dict, ...], ...]
    riichi: tuple[bool, ...]
    dora_indicators: tuple[str, ...]
    dealer: int
    round_wind: int
    kyoku: int
    honba: int
    riichi_sticks: int
    wall_remaining: int
    legal_actions: tuple[Action, ...]
    history: tuple[dict, ...] = ()
    game_id: str = ""
    mode: str = "4p-red-half"
    schema_version: str = SCHEMA_VERSION
    ruleset: str = RULESET

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.ruleset != RULESET:
            raise ValueError("Unsupported schema or ruleset")
        if not 0 <= self.perspective_seat < 4 or not 0 <= self.dealer < 4:
            raise ValueError("Invalid player seat")
        if any(len(x) != 4 for x in (self.scores, self.rivers, self.melds, self.riichi)):
            raise ValueError("Four-player state requires exactly four public player entries")
        if not self.legal_actions or len({a.id for a in self.legal_actions}) != len(
            self.legal_actions
        ):
            raise ValueError("Decision states require nonempty, unique legal candidates")
        if len(self.hand) > 14 or not 0 <= self.wall_remaining <= 70:
            raise ValueError("Invalid hand or wall size")
        visible = list(self.hand) + list(self.dora_indicators)
        for river in self.rivers:
            visible.extend(river)
        for melds in self.melds:
            for meld in melds:
                visible.extend(meld["tiles"])
        counts = Counter(Tile.parse(t).tile34 for t in visible)
        if any(n > 4 for n in counts.values()):
            raise ValueError("More than four visible copies of a tile")
        red_counts = Counter(Tile.parse(t).tile34 for t in visible if Tile.parse(t).is_red)
        if any(n > 1 for n in red_counts.values()):
            raise ValueError("More than one red five per suit")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> MahjongState:
        # Explicit whitelist: oracle fields and arbitrary JSON never enter policy tensors.
        known = {name: value[name] for name in cls.__dataclass_fields__ if name in value}
        for key in ("hand", "scores", "riichi", "dora_indicators", "history"):
            if key in known:
                known[key] = tuple(known[key])
        for key in ("rivers", "melds"):
            known[key] = tuple(tuple(x) for x in known[key])
        known["legal_actions"] = tuple(Action.from_dict(x) for x in known["legal_actions"])
        result = cls(**known)
        result.validate()
        return result


@dataclass
class Transition:
    episode_id: str
    decision_id: int
    seat: int
    state: MahjongState
    chosen_action: int
    next_state: MahjongState | None
    reward: float
    discount: float
    done: bool
    elapsed_steps: int = 1
    metadata: dict = field(default_factory=dict)

    def validate(self):
        self.state.validate()
        if self.next_state is not None:
            self.next_state.validate()
        if self.seat != self.state.perspective_seat:
            raise ValueError("Transition seat mismatch")
        if not 0 <= self.chosen_action < len(self.state.legal_actions):
            raise ValueError("Chosen action is not legal")
        if self.done != (self.next_state is None):
            raise ValueError("Terminal transitions must have no next state")
        if not 0 <= self.discount <= 1 or self.elapsed_steps < 1:
            raise ValueError("Invalid semi-MDP discount")
        if not math.isfinite(self.reward):
            raise ValueError("Transition reward must be finite")
        if self.next_state is not None and self.next_state.perspective_seat != self.seat:
            raise ValueError("Next state belongs to a different player")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> Transition:
        value = dict(value)
        value["state"] = MahjongState.from_dict(value["state"])
        if value["next_state"] is not None:
            value["next_state"] = MahjongState.from_dict(value["next_state"])
        result = cls(**value)
        result.validate()
        return result


def game_split(game_id: str, seed: int = 0) -> str:
    bucket = int(hashlib.sha256(f"{seed}:{game_id}".encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"
