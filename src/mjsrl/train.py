"""Behavior cloning, discrete IQL and full-episode PPO league self-play."""

from __future__ import annotations

import copy
import json
import math
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .checkpoint import load_checkpoint, restore_rng, save_checkpoint, seed_all
from .features import collate
from .model import ModelConfig, PolicyModel
from .rollout import HeuristicAgent, ModelAgent, play_game
from .storage import file_hash, read_transitions


@dataclass
class TrainConfig:
    algorithm: str = "bc"
    steps: int = 50
    batch_size: int = 16
    accumulation: int = 1
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    seed: int = 42
    device: str = "cpu"
    precision: str = "fp32"
    expectile: float = 0.7
    beta: float = 3.0
    weight_clip: float = 100.0
    target_tau: float = 0.005
    ppo_clip: float = 0.15
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    target_kl: float = 0.02
    ppo_epochs: int = 3
    games_per_update: int = 1
    gamma: float = 0.997
    mode: str = "4p-red-half"
    reward_mode: str = "combined"
    save_every: int = 10
    threads: int = 2
    tensorboard: bool = False

    def validate(self):
        if self.algorithm not in ("bc", "iql", "ppo"):
            raise ValueError("Unknown training algorithm")
        if (
            min(
                self.steps,
                self.batch_size,
                self.accumulation,
                self.save_every,
                self.threads,
                self.ppo_epochs,
                self.games_per_update,
            )
            < 1
        ):
            raise ValueError("Training sizes and intervals must be positive")
        if not 0 < self.expectile < 1 or not 0 <= self.target_tau <= 1:
            raise ValueError("Invalid IQL parameters")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("Invalid discount or GAE lambda")
        if self.learning_rate <= 0 or self.beta < 0 or self.weight_clip < 1:
            raise ValueError("Invalid learning rate or advantage weighting")
        if self.precision not in ("fp32", "bf16"):
            raise ValueError("Use fp32 or bf16 precision")
        if self.precision == "bf16" and (
            not self.device.startswith("cuda") or not torch.cuda.is_bf16_supported()
        ):
            raise ValueError("bf16 training requires a supported CUDA device")


class MetricsLogger:
    def __init__(self, output: Path, tensorboard=False):
        output.mkdir(parents=True, exist_ok=True)
        self.path = output / "metrics.jsonl"
        self.writer = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(str(output / "tensorboard"))

    def log(self, step, metrics):
        if not all(math.isfinite(v) for v in metrics.values()):
            raise FloatingPointError(f"Non-finite training metrics: {metrics}")
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"step": step, **metrics}) + "\n")
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)
        if step == 1 or step % 10 == 0:
            print(json.dumps({"step": step, **metrics}), flush=True)

    def close(self):
        if self.writer:
            self.writer.close()


def expectile_loss(diff, tau):
    return (torch.where(diff < 0, 1 - tau, tau) * diff.square()).mean()


def offline_loss(model, target, records, config):
    batch = collate([t.state for t in records], config.device, model.config.history_length)
    labels = torch.tensor([t.chosen_action for t in records], device=config.device)
    output = model(batch)
    nll = F.cross_entropy(output["logits"], labels, reduction="none")
    metrics = {"accuracy": float((output["logits"].argmax(-1) == labels).float().mean())}
    if config.algorithm == "bc":
        return nll.mean(), {**metrics, "bc_loss": float(nll.mean().detach())}
    selected = labels.unsqueeze(1)
    with torch.no_grad():
        target_out = target(batch)
        q = torch.minimum(target_out["q1"], target_out["q2"]).gather(1, selected).squeeze(1)
        nonterminal = [i for i, t in enumerate(records) if not t.done]
        next_values = torch.zeros(len(records), device=config.device)
        if nonterminal:
            next_batch = collate(
                [records[i].next_state for i in nonterminal],
                config.device,
                model.config.history_length,
            )
            next_values[nonterminal] = model(next_batch)["value"].float()
        rewards = torch.tensor([t.reward for t in records], device=config.device)
        discounts = torch.tensor([t.discount for t in records], device=config.device)
        backup = rewards + discounts * next_values
        adv = q - output["value"]
        weights = (config.beta * adv).clamp(max=math.log(config.weight_clip)).exp()
    v_loss = expectile_loss(q - output["value"], config.expectile)
    q1 = output["q1"].gather(1, selected).squeeze(1)
    q2 = output["q2"].gather(1, selected).squeeze(1)
    q_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)
    actor_loss = (weights * nll).mean()
    loss = v_loss + q_loss + actor_loss
    return loss, {
        **metrics,
        "q_loss": float(q_loss.detach()),
        "v_loss": float(v_loss.detach()),
        "actor_loss": float(actor_loss.detach()),
        "q_mean": float(q.mean()),
        "advantage_mean": float(adv.mean()),
        "weight_max": float(weights.max()),
    }


def gae(records, gae_lambda):
    advantages, returns = [0.0] * len(records), [0.0] * len(records)
    running = 0.0
    next_value = 0.0
    for i in reversed(range(len(records))):
        item = records[i]
        value = item.metadata["value"]
        if item.done:
            running, next_value = 0.0, 0.0
        delta = item.reward + item.discount * next_value - value
        running = delta + item.discount * gae_lambda**item.elapsed_steps * running
        advantages[i], returns[i] = running, running + value
        next_value = value
    return advantages, returns


def _ppo_update(model, optimizer, config, records):
    adv, returns = gae(records, config.gae_lambda)
    advantages = torch.tensor(adv, device=config.device)
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    returns = torch.tensor(returns, device=config.device)
    old_log = torch.tensor([t.metadata["log_prob"] for t in records], device=config.device)
    metrics, updates = {}, 0
    for _ in range(config.ppo_epochs):
        permutation = torch.randperm(len(records)).tolist()
        for start in range(0, len(records), config.batch_size):
            ids = permutation[start : start + config.batch_size]
            batch = collate(
                [records[i].state for i in ids], config.device, model.config.history_length
            )
            labels = torch.tensor([records[i].chosen_action for i in ids], device=config.device)
            with autocast(config):
                output = model(batch)
                dist = torch.distributions.Categorical(logits=output["logits"])
                log_prob = dist.log_prob(labels)
                logratio = log_prob - old_log[ids]
                ratio = logratio.exp()
                policy_loss = -torch.minimum(
                    ratio * advantages[ids],
                    ratio.clamp(1 - config.ppo_clip, 1 + config.ppo_clip) * advantages[ids],
                ).mean()
                value_loss = F.mse_loss(output["value"].float(), returns[ids])
                entropy = dist.entropy().mean()
                loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            kl = float(((ratio - 1) - logratio).mean().detach())
            metrics = {
                "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "entropy": float(entropy.detach()),
                "approx_kl": kl,
                "clip_fraction": float(
                    ((ratio - 1).abs() > config.ppo_clip).float().mean().detach()
                ),
            }
            updates += 1
            if kl > config.target_kl:
                return {**metrics, "minibatch_updates": float(updates)}
    return {**metrics, "minibatch_updates": float(updates)}


def autocast(config):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if config.precision == "bf16"
        else nullcontext()
    )


def train(
    config: TrainConfig,
    output: Path,
    dataset: Path | None = None,
    init: Path | None = None,
    resume: Path | None = None,
    model_config=ModelConfig(),
    split="train",
    validation: Path | None = None,
):
    config.validate()
    torch.set_num_threads(config.threads)
    seed_all(config.seed)
    if init and resume:
        raise ValueError("Choose initialization or continuation, not both")
    if config.algorithm != "ppo" and dataset is None:
        raise ValueError("BC/IQL requires a dataset")
    dataset_sha = file_hash(dataset) if dataset else None
    start, payload = 0, None
    if init or resume:
        model, payload = load_checkpoint(resume or init, config.device)
    else:
        model = PolicyModel(model_config).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    target = copy.deepcopy(model).eval()
    baseline = copy.deepcopy(model).cpu().eval()
    snapshots = []
    if resume:
        previous = payload["config"]
        for key, value in asdict(config).items():
            if key not in ("steps", "save_every", "tensorboard") and previous[key] != value:
                raise ValueError(f"Resume configuration changed: {key}")
        if payload["extra"].get("dataset_sha256") != dataset_sha:
            raise ValueError("Resume dataset changed")
        if payload["extra"].get("split") != split:
            raise ValueError("Resume data split changed")
        optimizer.load_state_dict(payload["optimizer"])
        start = payload["step"]
        if config.steps <= start:
            raise ValueError("Total steps must exceed the resumed checkpoint step")
        target.load_state_dict(payload["extra"]["target"])
        baseline.load_state_dict(payload["extra"]["baseline"])
        snapshots = payload["extra"].get("snapshots", [])
        restore_rng(payload["rng"])
    records = read_transitions(dataset, split) if dataset else None
    logger = MetricsLogger(output, config.tensorboard)
    extra = {"dataset_sha256": dataset_sha, "split": split}
    try:
        for step in range(start + 1, config.steps + 1):
            if config.algorithm == "ppo":
                learner_seat = (step - 1) % 4
                learner = ModelAgent(model, config.device)
                batch_records, ranks = [], []
                for game in range(config.games_per_update):
                    agents = {learner_seat: learner}
                    for seat in range(4):
                        if seat == learner_seat:
                            continue
                        branch = random.random()
                        if branch < 0.5:
                            opponent = copy.deepcopy(model).eval()
                        elif branch < 0.9 and snapshots:
                            opponent = PolicyModel(model.config).to(config.device)
                            pool = snapshots[-5:] if branch < 0.75 else snapshots
                            opponent.load_state_dict(random.choice(pool))
                        elif branch < 0.9:
                            opponent = copy.deepcopy(baseline).to(config.device)
                        else:
                            agents[seat] = HeuristicAgent(random.randrange(2**31))
                            continue
                        agents[seat] = ModelAgent(opponent, config.device)
                    result = play_game(
                        agents,
                        config.seed + step * config.games_per_update + game,
                        config.mode,
                        config.gamma,
                        config.reward_mode,
                    )
                    batch_records.extend(t for t in result.transitions if t.seat == learner_seat)
                    ranks.append(result.ranks[learner_seat])
                model.train()
                metrics = _ppo_update(model, optimizer, config, batch_records)
                metrics["rollout_mean_rank"] = sum(ranks) / len(ranks)
                metrics["decisions"] = float(len(batch_records))
                snapshots.append(
                    {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                )
                snapshots = snapshots[-10:]
            else:
                model.train()
                optimizer.zero_grad(set_to_none=True)
                metrics = {}
                for _ in range(config.accumulation):
                    sampled = random.choices(records, k=config.batch_size)
                    with autocast(config):
                        loss, values = offline_loss(model, target, sampled, config)
                    (loss / config.accumulation).backward()
                    for key, value in values.items():
                        metrics[key] = metrics.get(key, 0) + value / config.accumulation
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                with torch.no_grad():
                    for target_param, param in zip(
                        target.parameters(), model.parameters(), strict=True
                    ):
                        target_param.lerp_(param, config.target_tau)
            logger.log(step, metrics)
            if step % config.save_every == 0 or step == config.steps:
                extra.update(
                    target=target.state_dict(),
                    baseline=baseline.state_dict(),
                    snapshots=snapshots,
                    sampler_cursor=step * config.batch_size * config.accumulation,
                    parent_checkpoint=str(init) if init else None,
                )
                save_checkpoint(output / "latest.pt", model, optimizer, step, asdict(config), extra)
                if validation:
                    validation_metrics = evaluate_offline(model, validation, config.device)
                    logger.log(step, {f"validation_{k}": v for k, v in validation_metrics.items()})
        return model
    finally:
        logger.close()


@torch.no_grad()
def evaluate_offline(model, dataset, device="cpu", split="validation"):
    records = read_transitions(dataset, split)
    model.eval()
    total_loss, correct = 0.0, 0
    for start in range(0, len(records), 32):
        items = records[start : start + 32]
        output = model(collate([t.state for t in items], device, model.config.history_length))
        labels = torch.tensor([t.chosen_action for t in items], device=device)
        total_loss += float(F.cross_entropy(output["logits"], labels, reduction="sum"))
        correct += int((output["logits"].argmax(-1) == labels).sum())
    return {"nll": total_loss / len(records), "accuracy": correct / len(records)}
