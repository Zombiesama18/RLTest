"""Atomic checkpoints with optimizer, target model, sampler and complete RNG state."""

import importlib.metadata
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from .model import ModelConfig, PolicyModel
from .schema import RULESET, SCHEMA_VERSION
from .storage import file_hash


def seed_all(seed: int):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def save_checkpoint(path: Path, model, optimizer, step, config, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "format_version": 1,
        "schema_version": SCHEMA_VERSION,
        "ruleset": RULESET,
        "model_config": model.config_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": config,
        "rng": rng_state(),
        "extra": extra or {},
    }
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "ruleset": RULESET,
        "step": step,
        "model_config": model.config_dict(),
        "training_config": config,
        "parameters": sum(p.numel() for p in model.parameters()),
        "sha256": file_hash(path),
        "engine_version": importlib.metadata.version("riichienv"),
        "torch_version": torch.__version__,
        "arena_verified": False,
    }
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_checkpoint(path: Path, device="cpu"):
    # Trainer files contain Python/NumPy RNG state; load only your own trusted local files.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("format_version") != 1
        or payload["schema_version"] != SCHEMA_VERSION
        or payload["ruleset"] != RULESET
    ):
        raise ValueError("Checkpoint format, schema or ruleset mismatch")
    model = PolicyModel(ModelConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["model"])
    return model, payload
