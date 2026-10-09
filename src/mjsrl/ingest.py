"""Authorized decoded Mahjong Soul exports and MJAI replay datasets.

Binary protobuf and URLs are intentionally not accepted: decode/export them first.
Native replay steps, not a homemade rules replayer, supply expert labels and legality.
"""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

from riichienv import GameRule, MjaiReplay, MjSoulReplay

from .engine import canonical_action, snapshot, update_history
from .rollout import terminal_reward
from .schema import Transition
from .storage import file_hash, write_transitions

PERMISSION_BASES = {"user_owned", "explicit_permission", "self_generated"}
KNOWN_RECORDS = {
    "NewRound",
    "DealTile",
    "DiscardTile",
    "ChiPengGang",
    "AnGangAddGang",
    "Hule",
    "NoTile",
    "LiuJu",
    "dora",
}


def load_replay(path: Path, format: str):
    if format == "mjai":
        return MjaiReplay.from_jsonl(str(path), rule="mjsoul")
    if format != "majsoul":
        raise ValueError("Use majsoul or mjai format")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if isinstance(value, dict):
        rounds = value.get("data", value.get("rounds"))
    else:
        rounds = value
    if not isinstance(rounds, list) or not rounds:
        raise ValueError("Expected decoded export {data: [[{name, data}, ...], ...]}")
    for round_events in rounds:
        if not round_events or round_events[0].get("name") != "NewRound":
            raise ValueError("Every round must begin with NewRound")
        if len(round_events[0]["data"]["scores"]) != 4:
            raise ValueError("Only four-player records are accepted")
        for event in round_events:
            if event.get("name") not in KNOWN_RECORDS:
                raise ValueError(
                    f"Unsupported record (refusing silent omission): {event.get('name')}"
                )
    replay = MjSoulReplay.from_dict({"data": rounds})
    _, mismatches = replay.verify()
    if mismatches:
        raise ValueError(f"Replay scoring verification found {mismatches} mismatches")
    return replay


def replay_transitions(
    path: Path, format="mjai", gamma=0.997, reward_mode="combined", expected_scores=None
):
    replay = load_replay(path, format)
    game_id = "sha256:" + file_hash(path)
    rounds = list(replay.take_kyokus())
    if not rounds:
        raise ValueError("Replay has no rounds")
    initial = list(rounds[0].scores)
    final = rounds[-1].game_end_scores or rounds[-1].end_scores
    if expected_scores is not None and list(final) != list(expected_scores):
        raise ValueError(f"Golden replay settlement mismatch: {final} != {expected_scores}")
    # Stable seat-order tie break, matching the simulator's rank convention.
    order = sorted(range(4), key=lambda seat: (-final[seat], seat))
    ranks = {seat: rank for rank, seat in enumerate(order, 1)}
    histories, pending, clock = {s: [] for s in range(4)}, {}, 0
    for round_record in rounds:
        histories = {s: [] for s in range(4)}
        for seat, obs, chosen_native in round_record.steps(
            rule=GameRule.default_mjsoul(), skip_single_action=False
        ):
            histories[seat] = update_history(histories[seat], obs)
            if not obs.legal_actions():
                continue
            state, _ = snapshot(obs, histories[seat], game_id, "4p-red-half")
            chosen = canonical_action(chosen_native, obs, histories[seat])
            candidates = [
                i for i, action in enumerate(state.legal_actions) if action.id == chosen.id
            ]
            if not candidates:
                # Physical copy IDs can differ; compare semantic tiles while retaining red fives.
                candidates = [
                    i
                    for i, action in enumerate(state.legal_actions)
                    if (action.type, action.tile, action.consume)
                    == (chosen.type, chosen.tile, chosen.consume)
                ]
            if not candidates:
                raise ValueError(f"Replay label not legal at decision {clock}: {chosen}")
            if seat in pending:
                previous, index, old_clock = pending.pop(seat)
                elapsed = max(1, clock - old_clock)
                yield Transition(
                    game_id,
                    old_clock,
                    seat,
                    previous,
                    index,
                    state,
                    0,
                    gamma**elapsed,
                    False,
                    elapsed,
                    {"source": format},
                )
            pending[seat] = state, candidates[0], clock
            clock += 1
    for seat, (state, index, old_clock) in pending.items():
        elapsed = max(1, clock - old_clock)
        reward = terminal_reward(final[seat] - initial[seat], ranks[seat], reward_mode)
        yield Transition(
            game_id,
            old_clock,
            seat,
            state,
            index,
            None,
            reward,
            0,
            True,
            elapsed,
            {"source": format, "final_scores": list(final)},
        )


def ingest(path: Path, output: Path, manifest: Path, format="majsoul", gamma=0.997):
    metadata = json.loads(manifest.read_text(encoding="utf-8-sig"))
    if metadata.get("permission_basis") not in PERMISSION_BASES:
        raise ValueError("Manifest needs user_owned, explicit_permission or self_generated basis")
    if metadata.get("ruleset") != "majsoul_4p_standard":
        raise ValueError("Manifest must specify majsoul_4p_standard")
    expected_hash = metadata.get("sha256")
    actual_hash = file_hash(path)
    if expected_hash is not None and expected_hash != actual_hash:
        raise ValueError("Raw replay checksum mismatch")
    # Archive a copy without editing the source. Preserve identifier status accurately.
    vault = output.parent / "raw" / (actual_hash + "".join(path.suffixes))
    vault.parent.mkdir(parents=True, exist_ok=True)
    if not vault.exists():
        shutil.copyfile(path, vault)
    vault.with_suffix(vault.suffix + ".manifest.json").write_text(
        json.dumps({**metadata, "sha256": actual_hash, "source_filename": path.name}, indent=2),
        encoding="utf-8",
    )
    count = write_transitions(
        output,
        replay_transitions(
            path,
            format,
            gamma,
            expected_scores=metadata.get("expected_final_scores"),
        ),
    )
    return {"transitions": count, "raw_sha256": actual_hash, "output": str(output)}
