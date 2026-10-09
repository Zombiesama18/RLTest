"""Calibrated offline tile templates with confidence rejection and temporal fusion.

Numeric HUD fields and engine legal actions must come from an annotated replay state.
This module does not claim to reconstruct an entire table without a calibrated profile.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .schema import MahjongState, Tile


@dataclass
class StateEstimate:
    state: MahjongState | None
    overall_confidence: float
    stable: bool
    diagnostics: dict = field(default_factory=dict)


class OfflineVision:
    def __init__(self, template_dir: Path, threshold=0.95, margin=0.02, stable_frames=3):
        if not 0 <= threshold <= 1 or margin < 0 or stable_frames < 1:
            raise ValueError("Invalid vision confidence settings")
        self.threshold, self.margin = threshold, margin
        self.templates = {}
        for path in sorted(template_dir.glob("*.png")):
            label = str(Tile.parse(path.stem))
            with Image.open(path) as image:
                self.templates[label] = self._features(image)
        if len(self.templates) != 37:
            raise ValueError("Provide 37 templates: 1m..9s, 1z..7z and 5mr/5pr/5sr PNGs")
        self.memory = deque(maxlen=stable_frames)

    @staticmethod
    def _features(image):
        return np.asarray(image.convert("RGB").resize((32, 48)), dtype=np.float32) / 255

    def classify(self, crop):
        image = self._features(crop)
        ranked = sorted(
            (1 - float(np.abs(image - template).mean()), label)
            for label, template in self.templates.items()
        )
        best, label = ranked[-1]
        accepted = best >= self.threshold and best - ranked[-2][0] >= self.margin
        return label if accepted else None, best

    def recognize(
        self, frame: Path, profile: dict, annotated_state: dict, context="replay"
    ) -> StateEstimate:
        if context not in ("replay", "offline_recording"):
            raise ValueError("Vision requires an offline replay/recording context")
        fields, scores, unknown = {}, [], []
        with Image.open(frame) as image:
            width, height = image.size
            for field_name, boxes in profile["tile_slots"].items():
                values = []
                for index, box in enumerate(boxes):
                    if len(box) != 4 or not all(0 <= v <= 1 for v in box):
                        raise ValueError("ROI coordinates must be normalized x,y,w,h")
                    x, y, w, h = box
                    if w <= 0 or h <= 0 or x + w > 1 or y + h > 1:
                        raise ValueError("ROI is empty or outside the frame")
                    crop = image.crop(
                        (
                            round(x * width),
                            round(y * height),
                            round((x + w) * width),
                            round((y + h) * height),
                        )
                    )
                    label, score = self.classify(crop)
                    scores.append(score)
                    if label is None:
                        unknown.append(f"{field_name}.{index}")
                    values.append(label)
                fields[field_name] = values
        if unknown:
            self.memory.clear()
            return StateEstimate(
                None, min(scores, default=0), False, {"status": "WAIT", "uncertain_fields": unknown}
            )
        # Explicit supported field paths; arbitrary profile keys cannot inject policy data.
        state_data = dict(annotated_state)
        state_data["rivers"] = [list(r) for r in state_data["rivers"]]
        for name, values in fields.items():
            if name in ("hand", "dora_indicators"):
                state_data[name] = values
            elif name.startswith("rivers.") and name[-1] in "0123":
                state_data["rivers"][int(name[-1])] = values
            else:
                raise ValueError(f"Unsupported calibrated tile field: {name}")
        try:
            state = MahjongState.from_dict(state_data)
        except ValueError as error:
            self.memory.clear()
            return StateEstimate(
                None, min(scores, default=0), False, {"status": "WAIT", "rule_error": str(error)}
            )
        reference = MahjongState.from_dict(annotated_state)
        if (
            state.hand != reference.hand
            or state.rivers != reference.rivers
            or state.dora_indicators != reference.dora_indicators
        ):
            self.memory.clear()
            return StateEstimate(
                None,
                min(scores, default=0),
                False,
                {
                    "status": "WAIT",
                    "rule_error": "Recognized tiles changed; replay engine must "
                    "confirm the state and regenerate legal actions",
                },
            )
        signature = repr((state.hand, state.rivers, state.dora_indicators))
        self.memory.append(signature)
        stable = len(self.memory) == self.memory.maxlen and len(set(self.memory)) == 1
        return StateEstimate(
            state if stable else None,
            min(scores, default=0),
            stable,
            {
                "status": "READY" if stable else "WAIT",
                "recognized_fields": list(fields),
                "remaining_fields_source": "annotated_replay_state",
            },
        )
