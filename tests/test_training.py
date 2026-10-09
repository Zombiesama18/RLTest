from dataclasses import replace

import pytest
import torch

from mjsrl.checkpoint import load_checkpoint
from mjsrl.model import ModelConfig
from mjsrl.storage import write_transitions
from mjsrl.train import TrainConfig, expectile_loss, gae, train


@pytest.mark.parametrize("algorithm", ["bc", "iql"])
def test_offline_resume_exact(records, tmp_path, algorithm):
    data = tmp_path / "data.parquet"
    write_transitions(data, records)
    config = TrainConfig(algorithm=algorithm, steps=4, batch_size=4, accumulation=2, threads=1)
    small = ModelConfig(width=32, layers=1, heads=4, history_length=16)
    continuous = train(config, tmp_path / "continuous", data, model_config=small, split=None)
    train(replace(config, steps=2), tmp_path / "part", data, model_config=small, split=None)
    resumed = train(
        config, tmp_path / "resumed", data, resume=tmp_path / "part/latest.pt", split=None
    )
    for key, value in continuous.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[key]), key
    _, payload = load_checkpoint(tmp_path / "resumed/latest.pt")
    assert payload["step"] == 4
    assert payload["extra"]["sampler_cursor"] == 32


def test_ppo_resume_exact(tmp_path):
    config = TrainConfig(
        algorithm="ppo", steps=2, batch_size=16, ppo_epochs=1, mode="4p-red-single", threads=1
    )
    small = ModelConfig(width=32, layers=1, heads=4, history_length=16)
    continuous = train(config, tmp_path / "continuous", model_config=small, split=None)
    train(replace(config, steps=1), tmp_path / "part", model_config=small, split=None)
    resumed = train(config, tmp_path / "resumed", resume=tmp_path / "part/latest.pt", split=None)
    for key, value in continuous.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[key]), key


def test_expectile_sign():
    diff = torch.tensor([-1.0, 1.0])
    assert expectile_loss(diff, 0.7).item() == pytest.approx(0.5)
    assert expectile_loss(torch.tensor([1.0]), 0.7).item() == pytest.approx(0.7)
    assert expectile_loss(torch.tensor([-1.0]), 0.7).item() == pytest.approx(0.3)


def test_gae_terminal_reset(records):
    first = replace(records[0], reward=0, discount=0.5, elapsed_steps=1, metadata={"value": 0.0})
    last = replace(
        records[0],
        reward=2,
        discount=0,
        done=True,
        next_state=None,
        elapsed_steps=1,
        metadata={"value": 0.0},
    )
    adv, returns = gae([first, last, last], gae_lambda=1)
    assert adv == [1, 2, 2]
    assert returns == adv


def test_changed_data_resume_rejected(records, tmp_path):
    data = tmp_path / "data.parquet"
    write_transitions(data, records)
    train(TrainConfig(steps=1), tmp_path / "run", data, split=None)
    write_transitions(data, records[:-1])
    with pytest.raises(ValueError, match="dataset changed"):
        train(
            TrainConfig(steps=2),
            tmp_path / "next",
            data,
            resume=tmp_path / "run/latest.pt",
            split=None,
        )
