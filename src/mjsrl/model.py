"""Transformer state encoder with candidate policy, twin Q and value heads."""

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .features import GLOBAL_DIM
from .schema import ACTION_TYPES


@dataclass(frozen=True)
class ModelConfig:
    width: int = 64
    layers: int = 2
    heads: int = 4
    history_length: int = 64


class PolicyModel(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        d = config.width
        self.tile = nn.Embedding(38, d, padding_idx=0)
        self.kind = nn.Embedding(32, d)
        self.seat = nn.Embedding(5, d)
        self.flag = nn.Embedding(2, d)
        self.position = nn.Embedding(1024, d)
        self.numeric = nn.Sequential(nn.Linear(GLOBAL_DIM, d), nn.LayerNorm(d))
        layer = nn.TransformerEncoderLayer(
            d, config.heads, d * 4, dropout=0, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.action_kind = nn.Embedding(len(ACTION_TYPES), d)
        self.actor = self._head(d * 2, d)
        self.q1 = self._head(d * 2, d)
        self.q2 = self._head(d * 2, d)
        self.value = self._head(d, d)

    @staticmethod
    def _head(input_dim, d):
        return nn.Sequential(nn.Linear(input_dim, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, batch: dict) -> dict:
        t = batch["tokens"]
        if t.shape[1] + 1 > 1024:
            raise ValueError("Observation exceeds maximum 1024 tokens")
        emb = self.kind(t[..., 0]) + self.tile(t[..., 1])
        emb = emb + self.seat(t[..., 2]) + self.flag(t[..., 3])
        emb = torch.cat((self.numeric(batch["globals"]).unsqueeze(1), emb), dim=1)
        emb = emb + self.position(torch.arange(emb.shape[1], device=emb.device)).unsqueeze(0)
        padding = torch.cat(
            (torch.zeros(t.shape[0], 1, dtype=torch.bool, device=t.device), batch["padding"]), dim=1
        )
        state = self.norm(self.encoder(emb, src_key_padding_mask=padding)[:, 0])
        a = batch["actions"]
        action = self.action_kind(a[..., 0]) + self.tile(a[..., 1])
        action = action + self.tile(a[..., 2:6]).sum(-2)
        action = action + self.seat(a[..., 6]) + self.flag(a[..., 7])
        joined = torch.cat((state.unsqueeze(1).expand_as(action), action), dim=-1)
        logits = self.actor(joined).squeeze(-1)
        logits = logits.masked_fill(~batch["legal_mask"], -torch.inf)
        return {
            "logits": logits,
            "q1": self.q1(joined).squeeze(-1),
            "q2": self.q2(joined).squeeze(-1),
            "value": self.value(state).squeeze(-1),
        }

    def config_dict(self):
        return asdict(self.config)
