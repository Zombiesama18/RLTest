"""Matched-seed, seat-rotated evaluations with bootstrap CIs clustered by seed."""

import json
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_checkpoint
from .rollout import HeuristicAgent, ModelAgent, play_game


def bootstrap_ci(values, seed=42, repetitions=2000):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    # One bootstrap unit is an entire four-seat matched-seed block.
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(repetitions)]
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def arena(
    candidate: Path,
    baseline: Path | None,
    output: Path,
    seeds=10,
    first_seed=10000,
    mode="4p-red-half",
    device="cpu",
):
    if seeds < 1:
        raise ValueError("At least one seed block is required")
    torch.set_num_threads(2)
    model, _ = load_checkpoint(candidate, device)
    candidate_agent = ModelAgent(model, device, stochastic=False)
    base_model = load_checkpoint(baseline, device)[0] if baseline else None
    rows, rank_blocks, delta_blocks, score_blocks = [], [], [], []
    for seed in range(first_seed, first_seed + seeds):
        ranks, deltas, scores = [], [], []
        for seat in range(4):
            # Reset all baseline RNGs in both legs to keep paired conditions comparable.
            def baseline_agents():
                return {
                    s: ModelAgent(base_model, device, False)
                    if base_model
                    else HeuristicAgent(seed * 4 + s)
                    for s in range(4)
                }

            agents = baseline_agents()
            agents[seat] = candidate_agent
            result = play_game(agents, seed, mode)
            reference = play_game(baseline_agents(), seed, mode)
            ranks.append(result.ranks[seat])
            scores.append(result.scores[seat])
            deltas.append(result.ranks[seat] - reference.ranks[seat])
            hands = sum(e["type"] == "end_kyoku" for e in result.events)
            wins = sum(e["type"] == "hora" and e["actor"] == seat for e in result.events)
            deal_ins = sum(
                e["type"] == "hora" and e.get("target") == seat and e["actor"] != seat
                for e in result.events
            )
            rows.append(
                {
                    "seed": seed,
                    "seat": seat,
                    "rank": result.ranks[seat],
                    "score": result.scores[seat],
                    "baseline_rank": reference.ranks[seat],
                    "rank_delta": deltas[-1],
                    "hands": hands,
                    "wins": wins,
                    "deal_ins": deal_ins,
                }
            )
        rank_blocks.append(sum(ranks) / 4)
        delta_blocks.append(sum(deltas) / 4)
        score_blocks.append(sum(scores) / 4)
        print(json.dumps({"arena_seed": seed, "mean_rank": rank_blocks[-1]}), flush=True)
    hands = sum(row["hands"] for row in rows)
    report = {
        "candidate": str(candidate),
        "baseline": str(baseline) if baseline else "shanten",
        "mode": mode,
        "seed_blocks": seeds,
        "games": seeds * 8,
        "candidate_seat_games": seeds * 4,
        "mean_rank": float(np.mean(rank_blocks)),
        "rank_ci95": bootstrap_ci(rank_blocks),
        "paired_rank_delta": float(np.mean(delta_blocks)),
        "paired_rank_delta_ci95": bootstrap_ci(delta_blocks),
        "mean_score": float(np.mean(score_blocks)),
        "score_ci95": bootstrap_ci(score_blocks),
        "first_rate": sum(r["rank"] == 1 for r in rows) / len(rows),
        "fourth_rate": sum(r["rank"] == 4 for r in rows) / len(rows),
        "win_rate": sum(r["wins"] for r in rows) / max(1, hands),
        "deal_in_rate": sum(r["deal_ins"] for r in rows) / max(1, hands),
        "pairing": "same initial seed and seat; later rounds may diverge with game flow",
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
