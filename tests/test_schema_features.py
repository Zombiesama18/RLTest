import copy
from dataclasses import replace

import pytest
import torch

from mjsrl.features import collate
from mjsrl.model import PolicyModel
from mjsrl.schema import MahjongState, Tile, game_split


@pytest.mark.parametrize("tile", ["1m", "9p", "7z", "5mr", "5pr", "5sr"])
def test_tile_roundtrip(tile):
    assert str(Tile.parse(tile)) == tile


@pytest.mark.parametrize("tile", ["4mr", "0z", "8z", "10m", "?", "5mrr", "5zr"])
def test_invalid_tile_rejected(tile):
    with pytest.raises(ValueError):
        Tile.parse(tile)


def test_red_tiles_not_collapsed():
    assert Tile.parse("0m") == Tile.parse("5mr")
    assert Tile.from136(16) == Tile(4, True)
    assert Tile.from136(17) == Tile(4, False)
    assert Tile.parse("5m").tile37 != Tile.parse("5mr").tile37


def test_no_hidden_information_in_policy(state):
    public = state.to_dict()
    full = copy.deepcopy(public)
    full["oracle_state"] = {"hands": [["1m"] * 13] * 4, "wall": ["9s"] * 70}
    full["opponent_hands"] = [["7z"] * 13] * 3
    for event in full["history"]:
        event["tehais"] = [["1m"] * 13] * 4
        if event.get("type") == "tsumo" and event.get("actor") != state.perspective_seat:
            event["pai"] = "5mr"
    a = collate([MahjongState.from_dict(public)])
    b = collate([MahjongState.from_dict(full)])
    assert all(torch.equal(a[key], b[key]) for key in a)
    model = PolicyModel().eval()
    with torch.no_grad():
        assert torch.equal(model(a)["logits"], model(b)["logits"])


def test_candidate_order_and_padding(state):
    model = PolicyModel().eval()
    reversed_state = replace(state, legal_actions=tuple(reversed(state.legal_actions)))
    fewer = replace(state, legal_actions=state.legal_actions[:1])
    with torch.no_grad():
        logits = model(collate([state]))["logits"][0]
        reversed_logits = model(collate([reversed_state]))["logits"][0]
        padded = model(collate([state, fewer]))["logits"]
    assert torch.allclose(logits, reversed_logits.flip(0))
    assert torch.isneginf(padded[1, 1:]).all()
    assert torch.equal(padded[1].softmax(-1)[1:], torch.zeros_like(padded[1, 1:]))


def test_multiplicity_guard(state):
    invalid = replace(state, hand=("1m",) * 5)
    with pytest.raises(ValueError, match="four visible"):
        invalid.validate()


def test_split_is_whole_game():
    first = game_split("same-episode")
    assert all(game_split("same-episode") == first for _ in range(100))
    assert len({game_split(str(i)) for i in range(100)}) == 3
