"""Streaming sanity checks on temperature and humidity.

The models score each reading on its own, so they miss faults that only show up over time
or sit outside anything they were trained on. These rules watch the stream instead:
  range     value outside what normal operation has ever produced (plus a margin)
  jump      change between consecutive readings bigger than normal operation has shown
  flatline  both sensors repeat the exact same value for longer than normal operation has
Limits are learned from the normal training rows in train.py.
"""

import pandas as pd

SENSORS = ("Temperature", "Humidity")
RANGE_MARGIN = {"Temperature": 2.0, "Humidity": 5.0}
JUMP_FACTOR = 1.2
FLATLINE_FACTOR = 2


def learn_limits(normal_rows: pd.DataFrame) -> dict:
    """normal_rows must be in time order."""
    limits = {}
    for s in SENSORS:
        values = normal_rows[s]
        limits[s] = {
            "min": float(values.min() - RANGE_MARGIN[s]),
            "max": float(values.max() + RANGE_MARGIN[s]),
            "max_jump": float(values.diff().abs().max() * JUMP_FACTOR),
        }
    changed = (normal_rows[list(SENSORS)].diff() != 0).any(axis=1).cumsum()
    limits["flatline_readings"] = int(normal_rows.groupby(changed).size().max() * FLATLINE_FACTOR)
    return limits


class SensorMonitor:
    def __init__(self, limits: dict):
        self.limits = limits
        self.reset()

    def reset(self):
        self._previous = None
        self._repeats = 0

    def update(self, reading: dict) -> list[str]:
        """Return the reasons this reading looks faulty (empty list if none)."""
        reasons = []
        for s in SENSORS:
            value, lim = reading[s], self.limits[s]
            if not lim["min"] <= value <= lim["max"]:
                reasons.append(f"{s} out of range")
            if self._previous is not None and abs(value - self._previous[s]) > lim["max_jump"]:
                reasons.append(f"{s} jump")

        current = tuple(reading[s] for s in SENSORS)
        if self._previous is not None and current == tuple(self._previous[s] for s in SENSORS):
            self._repeats += 1
        else:
            self._repeats = 1
        if self._repeats > self.limits["flatline_readings"]:
            reasons.append("sensors flatlined")

        self._previous = {s: reading[s] for s in SENSORS}
        return reasons
