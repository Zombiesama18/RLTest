import json

import pytest

from mjsrl.engine import RiichiEnvAdapter
from mjsrl.ingest import ingest, replay_transitions
from mjsrl.rollout import HeuristicAgent, RandomAgent, play_game
from mjsrl.storage import file_hash, read_transitions, write_transitions


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("format,suffix", [("mjai", "jsonl"), ("majsoul", "majsoul.json")])
def test_golden_replay(fixture_dir, seed, format, suffix):
    manifest = json.loads((fixture_dir / f"seed_{seed}.manifest.json").read_text())
    rows = list(
        replay_transitions(
            fixture_dir / f"seed_{seed}.{suffix}",
            format,
            expected_scores=manifest["expected_final_scores"],
        )
    )
    assert rows
    assert sum(row.done for row in rows) == 4
    for row in rows:
        row.validate()
        if row.next_state is not None:
            assert row.seat == row.next_state.perspective_seat
    for seat in range(4):
        indices = [r.decision_id for r in rows if r.seat == seat]
        assert indices == sorted(indices)


def test_same_seed_same_wall_and_result():
    first, second = RiichiEnvAdapter("4p-red-single"), RiichiEnvAdapter("4p-red-single")
    assert first.reset(42) == second.reset(42)
    games = [
        play_game({s: HeuristicAgent(s) for s in range(4)}, 42, "4p-red-single") for _ in range(2)
    ]
    assert games[0].events == games[1].events
    assert games[0].transitions == games[1].transitions


def test_illegal_candidate_rejected():
    from dataclasses import replace

    env = RiichiEnvAdapter("4p-red-single")
    states = env.reset(42)
    chosen = {s: replace(state.legal_actions[0], id="invalid") for s, state in states.items()}
    with pytest.raises(ValueError, match="Illegal action"):
        env.step(chosen)


@pytest.mark.parametrize("seed", range(100))
def test_hundred_simulator_games(seed):
    result = play_game({s: RandomAgent(seed * 4 + s) for s in range(4)}, seed, "4p-red-single")
    assert sorted(result.ranks) == [1, 2, 3, 4]
    assert result.steps > 0
    for row in result.transitions:
        row.validate()


def test_ingest_and_parquet_roundtrip(fixture_dir, tmp_path):
    raw = fixture_dir / "seed_42.majsoul.json"
    before = file_hash(raw)
    output = tmp_path / "silver.parquet"
    result = ingest(raw, output, fixture_dir / "seed_42.manifest.json")
    assert result["transitions"] == len(read_transitions(output))
    assert file_hash(raw) == before
    assert output.with_suffix(".manifest.json").exists()


def test_reject_missing_permission(fixture_dir, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"permission_basis": "unknown"}')
    with pytest.raises(ValueError, match="Manifest needs"):
        ingest(fixture_dir / "seed_42.majsoul.json", tmp_path / "data.parquet", manifest)


def test_reject_wrong_golden_settlement(fixture_dir):
    with pytest.raises(ValueError, match="settlement mismatch"):
        list(replay_transitions(fixture_dir / "seed_42.jsonl", expected_scores=[0] * 4))


def test_reject_empty_dataset(tmp_path):
    with pytest.raises(ValueError, match="No transitions"):
        write_transitions(tmp_path / "empty.parquet", [])
    assert not (tmp_path / "empty.parquet").exists()
