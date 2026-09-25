"""Generate normal device readings from real normal traffic the models have not seen."""

import numpy as np
import pandas as pd

from .features import FEATURES

NETWORK_FIELDS = ["duration", "destination_bytes", "missed_bytes", "dst_ip_bytes", "Status"]
CLIMATES = ("recorded", "synthetic")


class NormalGenerator:
    """Produces normal readings from the test split's normal rows.

    climate="recorded"   walk through the normal rows in their recorded order, so the
                         temperature/humidity follow the real room. Honest baseline.
    climate="synthetic"  network fields sampled from normal rows; temperature and humidity
                         follow a mean-reverting walk between real (T, H) pairs. The walk
                         passes through climates the model never saw for normal traffic,
                         which exposes how much it relies on temperature and humidity.
    """

    def __init__(self, reference: pd.DataFrame, seed=None, climate: str = "recorded", retarget_every: int = 150):
        if climate not in CLIMATES:
            raise ValueError(f"climate must be one of {CLIMATES}")
        normals = reference[(reference["split"] == "test") & (reference["Label"] == 0)].sort_values("row_id")
        if normals.empty:
            raise ValueError("Reference data has no normal test rows")
        self.rng = np.random.default_rng(seed)
        self.climate = climate
        self._rows = normals[FEATURES].to_dict("records")
        self._pos = int(self.rng.integers(len(self._rows)))
        self._network = normals[NETWORK_FIELDS].to_dict("records")
        self._climate = normals[["Temperature", "Humidity"]].to_numpy()
        self.temp_bounds = tuple(np.quantile(self._climate[:, 0], [0.01, 0.99]))
        self.hum_bounds = tuple(np.quantile(self._climate[:, 1], [0.01, 0.99]))
        self.retarget_every = retarget_every
        self._steps = 0
        self._target = self._draw_target()
        self.temperature, self.humidity = self._target

    def next(self) -> dict:
        return self._recorded() if self.climate == "recorded" else self._synthetic()

    def _recorded(self) -> dict:
        reading = dict(self._rows[self._pos % len(self._rows)])
        self._pos += 1
        return reading

    def _draw_target(self):
        return self._climate[self.rng.integers(len(self._climate))].astype(float)

    def _synthetic(self) -> dict:
        if self._steps and self._steps % self.retarget_every == 0:
            self._target = self._draw_target()
        self._steps += 1

        target_t, target_h = self._target
        self.temperature += 0.05 * (target_t - self.temperature) + self.rng.normal(0, 0.1)
        self.humidity += 0.05 * (target_h - self.humidity) + self.rng.normal(0, 0.3)
        self.temperature = float(np.clip(self.temperature, *self.temp_bounds))
        self.humidity = float(np.clip(self.humidity, *self.hum_bounds))

        reading = dict(self._network[self.rng.integers(len(self._network))])
        reading["Temperature"] = round(self.temperature, 1)
        reading["Humidity"] = round(self.humidity, 1)
        return reading
