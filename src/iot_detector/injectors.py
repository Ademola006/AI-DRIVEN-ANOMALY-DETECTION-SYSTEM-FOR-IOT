"""Anomaly injectors: turn a normal reading into an attacked or faulty one.

replay_*  use real attack rows from the test split (what the model was trained to find).
dos, mitm are synthetic: they change only the fields the attack plausibly affects, keeping
          the room's current temperature and humidity, so they test whether the model has
          learned the attack itself rather than the conditions the attacks were recorded in.
spike, drift, stuck are sensor faults that are not in the dataset at all.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import FEATURES


@dataclass(frozen=True)
class InjectorSpec:
    label: str
    truth_type: str  # "dos", "mitm" or "fault"
    default_length: int
    description: str


INJECTORS = {
    "replay_dos": InjectorSpec("Replay DoS", "dos", 20, "Real DoS rows from the test split"),
    "replay_mitm": InjectorSpec("Replay MITM", "mitm", 20, "Real MITM rows from the test split"),
    "dos": InjectorSpec("Synthetic DoS", "dos", 20, "Long connection durations, device ON"),
    "mitm": InjectorSpec("Synthetic MITM", "mitm", 20, "Spoofed sensor values plus retransmitted bytes"),
    "spike": InjectorSpec("Sensor spike", "fault", 1, "One reading jumps far from the current value"),
    "drift": InjectorSpec("Sensor drift", "fault", 60, "Temperature creeps up, humidity creeps down"),
    "stuck": InjectorSpec("Sensor stuck", "fault", 40, "Temperature and humidity freeze"),
}

TEMP_LIMITS = (-10.0, 60.0)
HUM_LIMITS = (0.0, 100.0)


@dataclass
class Injection:
    event_id: int
    kind: str
    intensity: float
    length: int
    start_step: int
    step: int = 0
    state: dict = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return self.step < self.length

    @property
    def truth_type(self) -> str:
        return INJECTORS[self.kind].truth_type


class InjectorBank:
    def __init__(self, reference: pd.DataFrame, seed=None):
        self.rng = np.random.default_rng(seed)
        test = reference[reference["split"] == "test"]
        self._replay = {t: test.loc[test["Type"] == t, FEATURES].to_dict("records") for t in ("dos", "mitm")}
        train_dos = reference[(reference["split"] == "train") & (reference["Type"] == "dos")]
        self._dos_durations = train_dos["duration"].to_numpy()
        self._next_id = 0

    def start(self, kind: str, intensity: float = 1.0, length: int | None = None, start_step: int = 0) -> Injection:
        if kind not in INJECTORS:
            raise ValueError(f"Unknown injector {kind!r}; choose from {sorted(INJECTORS)}")
        self._next_id += 1
        return Injection(
            event_id=self._next_id,
            kind=kind,
            intensity=float(np.clip(intensity, 0.0, 1.0)),
            length=length or INJECTORS[kind].default_length,
            start_step=start_step,
        )

    def apply(self, injection: Injection, reading: dict) -> dict:
        out = dict(reading)
        getattr(self, f"_{injection.kind}")(injection, out)
        out["Temperature"] = round(float(np.clip(out["Temperature"], *TEMP_LIMITS)), 1)
        out["Humidity"] = round(float(np.clip(out["Humidity"], *HUM_LIMITS)), 1)
        injection.step += 1
        return out

    def _replay_dos(self, inj, r):
        r.update(self._replay["dos"][self.rng.integers(len(self._replay["dos"]))])

    def _replay_mitm(self, inj, r):
        r.update(self._replay["mitm"][self.rng.integers(len(self._replay["mitm"]))])

    def _dos(self, inj, r):
        attack_duration = self.rng.choice(self._dos_durations)
        r["duration"] = int(r["duration"] + inj.intensity * (attack_duration - r["duration"]))
        r["Status"] = "ON"
        r["destination_bytes"] = 51

    def _mitm(self, inj, r):
        # The attacker reports false sensor values and the flow shows retransmitted bytes.
        r["Temperature"] -= inj.intensity * self.rng.uniform(1.5, 3.0)
        r["Humidity"] -= inj.intensity * self.rng.uniform(4.0, 8.0)
        r["missed_bytes"] = 51
        r["destination_bytes"] = 51
        r["dst_ip_bytes"] = 248
        r["Status"] = "ON"

    def _spike(self, inj, r):
        sign = self.rng.choice([-1, 1])
        r["Temperature"] += sign * (4 + 8 * inj.intensity)
        r["Humidity"] += sign * (10 + 15 * inj.intensity)

    def _drift(self, inj, r):
        r["Temperature"] += (inj.step + 1) * 0.2 * inj.intensity
        r["Humidity"] -= (inj.step + 1) * 0.5 * inj.intensity

    def _stuck(self, inj, r):
        frozen = inj.state.setdefault("frozen", (r["Temperature"], r["Humidity"]))
        r["Temperature"], r["Humidity"] = frozen
