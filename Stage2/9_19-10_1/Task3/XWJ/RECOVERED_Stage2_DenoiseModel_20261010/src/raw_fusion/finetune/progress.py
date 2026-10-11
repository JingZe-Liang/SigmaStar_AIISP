"""ETA progress tracking for the A0 build.

Contract style follows the mainline reliable_detail_progress.py: a frozen
state dataclass with a protocol tag, a strict status state machine, an EWMA
seconds-per-unit estimate, and atomic progress.json publication.
"""

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

PROTOCOL = "finetune_a0_progress_v1"


@dataclass(frozen=True, slots=True)
class A0ProgressState:
    protocol: str
    status: str
    started_at: str
    updated_at: str
    completed_units: int
    total_units: int
    current_item: str | None
    current_stage: str | None
    elapsed_seconds: float
    seconds_per_unit_ewma: float | None
    eta_seconds: float | None
    estimated_finish_at: str | None
    eta_stable: bool


class A0ProgressTracker:
    def __init__(
        self,
        destination: Path,
        total_units: int,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if total_units <= 0:
            raise ValueError("total_units must be positive")
        self.destination = destination
        self.total_units = total_units
        self._monotonic = monotonic
        self._now = now
        self._started_monotonic: float | None = None
        self._started_at: str | None = None
        self._state: A0ProgressState | None = None
        self._unit_started_monotonic: float | None = None
        self._ewma: float | None = None

    @property
    def state(self) -> A0ProgressState:
        if self._state is None:
            raise RuntimeError("tracker has not started")
        return self._state

    def start(self) -> A0ProgressState:
        if self._state is not None:
            raise RuntimeError("tracker has already started")
        self._started_monotonic = self._monotonic()
        self._started_at = self._now().isoformat()
        return self._publish(status="calibrating", current_item=None, current_stage=None)

    def start_unit(self, item: str, stage: str) -> A0ProgressState:
        self._require_status("calibrating", "running")
        if self._unit_started_monotonic is not None:
            raise RuntimeError("a unit is already running")
        if self._state is None or self._state.completed_units >= self.total_units:
            raise RuntimeError("all units are already complete")
        self._unit_started_monotonic = self._monotonic()
        return self._publish(status="running", current_item=item, current_stage=stage)

    def complete_unit(self) -> A0ProgressState:
        self._require_status("running")
        if self._unit_started_monotonic is None or self._state is None:
            raise RuntimeError("there is no current unit")
        duration = self._monotonic() - self._unit_started_monotonic
        if duration < 0:
            raise ValueError("monotonic clock moved backwards")
        self._ewma = duration if self._ewma is None else 0.2 * duration + 0.8 * self._ewma
        completed = self._state.completed_units + 1
        self._unit_started_monotonic = None
        return self._publish(
            status="running",
            current_item=None,
            current_stage=None,
            completed_units=completed,
        )

    def complete(self) -> A0ProgressState:
        self._require_status("running")
        if self._state is None or self._state.completed_units != self.total_units:
            raise RuntimeError("cannot complete before all units are done")
        return self._publish(status="complete", current_item=None, current_stage=None)

    def failed(self) -> A0ProgressState:
        self._require_status("calibrating", "running")
        if self._state is None:
            raise RuntimeError("tracker has not started")
        self._unit_started_monotonic = None
        return self._publish(status="failed", current_item=None, current_stage=None)

    def _require_status(self, *allowed: str) -> None:
        if self._state is None:
            raise RuntimeError("tracker has not started")
        if self._state.status not in allowed:
            raise RuntimeError(f"invalid transition from {self._state.status}")

    def _publish(
        self,
        *,
        status: str,
        current_item: str | None,
        current_stage: str | None,
        completed_units: int | None = None,
    ) -> A0ProgressState:
        if self._started_monotonic is None or self._started_at is None:
            raise RuntimeError("tracker has not started")
        updated_at = self._now()
        elapsed = self._monotonic() - self._started_monotonic
        if elapsed < 0:
            raise ValueError("monotonic clock moved backwards")
        completed = self._state.completed_units if self._state else 0
        if completed_units is not None:
            completed = completed_units
        remaining = self.total_units - completed
        eta = self._ewma * remaining if self._ewma is not None else None
        if status == "complete":
            estimated_finish_at = updated_at.isoformat()
            eta = 0.0
        elif eta is not None:
            estimated_finish_at = datetime.fromtimestamp(
                updated_at.timestamp() + eta, tz=timezone.utc
            ).isoformat()
        else:
            estimated_finish_at = None
        state = A0ProgressState(
            protocol=PROTOCOL,
            status=status,
            started_at=self._started_at,
            updated_at=updated_at.isoformat(),
            completed_units=completed,
            total_units=self.total_units,
            current_item=current_item,
            current_stage=current_stage,
            elapsed_seconds=elapsed,
            seconds_per_unit_ewma=self._ewma,
            eta_seconds=eta,
            estimated_finish_at=estimated_finish_at,
            eta_stable=completed >= 5,
        )
        self._write(state)
        self._state = state
        return state

    def _write(self, state: A0ProgressState) -> None:
        self.destination.mkdir(parents=True, exist_ok=True)
        target = self.destination / "progress.json"
        handle, temporary = tempfile.mkstemp(prefix=".progress.", dir=self.destination)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(asdict(state), stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
