"""Simulation loop: produce readings, inject anomalies, score them, keep a scoreboard.

Two modes:
  live    normal readings from NormalGenerator; anomalies only appear when injected.
  replay  the held-out test rows in their original order, with their real labels.
Injections work in both modes and replace the reading while they are active.
"""

from collections import deque
from itertools import islice
from pathlib import Path

import pandas as pd

from . import MODEL_DIR
from .detector import Detector, StreamDetector
from .features import FEATURES
from .generator import NormalGenerator
from .injectors import INJECTORS, InjectorBank, Injection

MODES = ("live", "replay")


class Simulator:
    """max_history caps the stored readings (None = keep all); the scoreboard always covers the whole run."""

    def __init__(
        self,
        detector: StreamDetector | None = None,
        model_dir=MODEL_DIR,
        mode: str = "live",
        climate: str = "recorded",
        seed=None,
        reference: pd.DataFrame | None = None,
        max_history: int | None = None,
    ):
        self.detector = detector or StreamDetector(Detector(model_dir))
        self.reference = pd.read_csv(Path(model_dir) / "reference.csv") if reference is None else reference
        self.seed = seed
        self.climate = climate
        self.mode = mode
        self.max_history = max_history
        self.reset()

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str):
        if value not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self._mode = value

    def reset(self):
        """Clear history and restart the data sources. Keeps mode, climate and detector settings."""
        self.generator = NormalGenerator(self.reference, seed=self.seed, climate=self.climate)
        self.injectors = InjectorBank(self.reference, seed=self.seed)
        self.detector.reset()
        self._replay_rows = self.reference[self.reference["split"] == "test"].sort_values("row_id")
        self._replay_pos = 0
        self.injection: Injection | None = None
        self.events: list[dict] = []
        self.history: deque[dict] = deque(maxlen=self.max_history)
        self._counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        self.t = 0

    def inject(self, kind: str, intensity: float = 1.0, length: int | None = None) -> Injection:
        """Start an injection on the next step. Replaces any injection still running."""
        self.injection = self.injectors.start(kind, intensity, length, start_step=self.t)
        self.events.append(
            {
                "event_id": self.injection.event_id,
                "kind": kind,
                "label": INJECTORS[kind].label,
                "intensity": self.injection.intensity,
                "start": self.t,
                "length": self.injection.length,
                "detected_at": None,
            }
        )
        return self.injection

    def _base_reading(self) -> tuple[dict, int, str]:
        if self.mode == "live":
            return self.generator.next(), 0, "normal"
        row = self._replay_rows.iloc[self._replay_pos % len(self._replay_rows)]
        self._replay_pos += 1
        return {f: row[f] for f in FEATURES}, int(row["Label"]), str(row["Type"])

    def step(self) -> dict:
        reading, truth_label, truth_type = self._base_reading()
        event_id, kind = None, None
        if self.injection is not None and self.injection.active:
            reading = self.injectors.apply(self.injection, reading)
            truth_label, truth_type = 1, self.injection.truth_type
            event_id, kind = self.injection.event_id, self.injection.kind

        result = self.detector.score(reading)
        record = {
            "t": self.t,
            **reading,
            "truth_label": truth_label,
            "truth_type": truth_type,
            "event_id": event_id,
            "injected": kind,
            **{k: v for k, v in result.items() if k != "type_proba"},
        }
        if event_id is not None and result["anomaly"]:
            event = self.events[-1]
            if event["detected_at"] is None:
                event["detected_at"] = self.t

        truth, pred = truth_label == 1, result["anomaly"]
        self._counts[("t" if truth == pred else "f") + ("p" if pred else "n")] += 1
        self.history.append(record)
        self.t += 1
        return record

    def run(self, n: int) -> pd.DataFrame:
        for _ in range(n):
            self.step()
        return self.history_frame()

    def recent(self, n: int) -> list[dict]:
        """The last n readings, oldest first."""
        start = max(0, len(self.history) - n)
        return list(islice(self.history, start, None))

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame(list(self.history))

    def events_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.events, columns=["event_id", "kind", "label", "intensity", "start", "length", "detected_at"])
        df["latency"] = df["detected_at"] - df["start"]
        df["detected"] = df["detected_at"].notna()
        return df

    def scoreboard(self) -> dict:
        """Per-reading confusion counts plus per-event detection results."""
        tp, fp, tn, fn = (self._counts[k] for k in ("tp", "fp", "tn", "fn"))
        events = [e for e in self.events if e["start"] < self.t]
        detected = [e for e in events if e["detected_at"] is not None]
        return {
            "readings": self.t,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "false_alarm_rate": fp / (fp + tn) if fp + tn else None,
            "events": len(events),
            "events_detected": len(detected),
            "mean_latency": (sum(e["detected_at"] - e["start"] for e in detected) / len(detected)) if detected else None,
        }
