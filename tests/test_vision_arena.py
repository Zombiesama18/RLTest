import numpy as np
import pytest
from PIL import Image

from mjsrl.arena import bootstrap_ci
from mjsrl.schema import Tile
from mjsrl.vision import OfflineVision


def test_cluster_ci_and_insufficient_sample():
    assert bootstrap_ci([2.5]) is None
    assert bootstrap_ci([2, 2, 2]) == [2, 2]
    assert bootstrap_ci([1, 2, 3]) == bootstrap_ci([1, 2, 3])


def test_templates_reject_unknown_and_wait(tmp_path, state):
    rng = np.random.default_rng(42)
    tiles = [str(Tile(i)) for i in range(34)] + ["5mr", "5pr", "5sr"]
    images = {}
    for tile in tiles:
        image = Image.fromarray(rng.integers(0, 256, (48, 32, 3), dtype=np.uint8))
        image.save(tmp_path / f"{tile}.png")
        images[tile] = image
    recognizer = OfflineVision(tmp_path)
    label, score = recognizer.classify(images["5mr"])
    assert label == "5mr" and score == 1
    label, _ = recognizer.classify(Image.new("RGB", (32, 48), "black"))
    assert label is None
    hand = state.hand
    frame = Image.new("RGB", (len(hand) * 32, 48))
    for i, tile in enumerate(hand):
        frame.paste(images[tile], (i * 32, 0))
    path = tmp_path / "frame.png"
    frame.save(path)
    profile = {
        "tile_slots": {"hand": [[i / len(hand), 0, 1 / len(hand), 1] for i in range(len(hand))]}
    }
    first = recognizer.recognize(path, profile, state.to_dict())
    assert not first.stable and first.state is None
    recognizer.recognize(path, profile, state.to_dict())
    third = recognizer.recognize(path, profile, state.to_dict())
    assert third.stable and third.state.hand == hand
    stale = state.to_dict()
    stale["hand"] = list(reversed(hand))
    changed = recognizer.recognize(path, profile, stale)
    assert changed.state is None and not changed.stable
    assert "regenerate legal actions" in changed.diagnostics["rule_error"]
    with pytest.raises(ValueError, match="offline"):
        recognizer.recognize(path, profile, state.to_dict(), context="live_ranked")
