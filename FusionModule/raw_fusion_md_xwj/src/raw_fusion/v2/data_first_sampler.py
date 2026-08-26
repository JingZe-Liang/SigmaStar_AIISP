"""In-memory deterministic cell schedule for data-first V2."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from .data_first_dataset import DataFirstSampleRow


@dataclass(frozen=True, slots=True)
class DataFirstSchedule:
    rows: tuple[DataFirstSampleRow, ...]
    seed: int
    batch_size: int

    @property
    def digest(self) -> str:
        payload = json.dumps(
            {
                "seed": self.seed,
                "batch_size": self.batch_size,
                "rows": [row.__dict__ if hasattr(row, "__dict__") else (row.condition, row.split, row.source_frame, row.cell_y, row.cell_x) for row in self.rows],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()

    def resume(self, epoch: int, global_step: int) -> tuple[DataFirstSampleRow, ...]:
        start = (int(epoch) * len(self.rows)) + int(global_step) * self.batch_size
        return self.rows[start : start + self.batch_size]


class DeterministicCellSampler:
    @staticmethod
    def build(dataset, *, seed: int, batch_size: int, conditions: tuple[str, ...] = ("128x", "645x")) -> DataFirstSchedule:
        if batch_size <= 0 or batch_size % 2:
            raise ValueError("data-first batch size must be positive and even")
        if tuple(conditions) != ("128x", "645x"):
            raise ValueError("data-first sampler conditions are fixed to 128x and 645x")
        rows: list[DataFirstSampleRow] = []
        cells_y, cells_x = dataset.cell_shape
        def supported(cell_index: int) -> bool:
            cell_y, cell_x = divmod(cell_index, cells_x)
            origin_y = min(max(cell_y * 32 - 64, 0), 540 - 320)
            origin_x = min(max(cell_x * 32 - 64, 0), 960 - 320)
            return 0 <= (cell_y * 32 - (origin_y + 32)) // 32 < 8 and 0 <= (cell_x * 32 - (origin_x + 32)) // 32 < 8
        frame_lists = {condition: tuple(dataset.target_frames[condition]) for condition in conditions}
        for frame_index in range(max(len(frame_lists["128x"]), len(frame_lists["645x"]))):
            frames = {condition: frame_lists[condition][frame_index] for condition in conditions if frame_index < len(frame_lists[condition])}
            if len(frames) != len(conditions):
                continue
            # With cached supervision, create condition-paired positive and
            # negative batches. This prevents the safe q=0 class dominating.
            for class_id in (1, 0):
                selected: list[tuple[str, int]] = []
                for slot, condition in enumerate(conditions):
                    frame = frames[condition]
                    if hasattr(dataset, "policy_alpha_class"):
                        grid = dataset.policy_alpha_class(condition, frame)
                        candidates = [index for index, value in enumerate(grid.reshape(-1)) if int(value) == class_id and supported(index)]
                        if not candidates:
                            selected = []
                            break
                        cell_index = candidates[(int(seed) + frame_index * 37 + slot * 19 + class_id) % len(candidates)]
                    else:
                        available = [index for index in range(cells_y * cells_x) if supported(index)]
                        cell_index = available[(int(seed) + frame_index * 37 + slot * 19 + class_id) % len(available)]
                    selected.append((condition, cell_index))
                for condition, cell_index in selected:
                    rows.append(DataFirstSampleRow(condition, "train", frames[condition], cell_index // cells_x, cell_index % cells_x))
        return DataFirstSchedule(tuple(rows), int(seed), int(batch_size))


__all__ = ["DataFirstSchedule", "DeterministicCellSampler"]
