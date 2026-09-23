"""Load the saved models and score readings."""

import json
from pathlib import Path

import joblib
import numpy as np

from . import MODEL_DIR
from .features import to_frame
from .sensor_checks import SensorMonitor


class Detector:
    """Stateless scoring with the three saved models. Safe to share between sessions.

    p_anomaly        binary Random Forest (knows DoS and MITM); thresholded by StreamDetector.
    attack_type      3-class Random Forest: normal, dos or mitm.
    novelty_flag     Isolation Forest trained on normal traffic only: "unlike normal".
    """

    def __init__(self, model_dir=MODEL_DIR):
        model_dir = Path(model_dir)
        self.metadata = json.loads((model_dir / "metadata.json").read_text())
        self.binary = joblib.load(model_dir / "binary_rf.joblib")
        self.attack_type = joblib.load(model_dir / "attack_type_rf.joblib")
        self.novelty = joblib.load(model_dir / "novelty_iforest.joblib")
        self.default_threshold = self.metadata["binary_threshold"]
        self._anomaly_col = list(self.binary.classes_).index(1)
        self._type_classes = [str(c) for c in self.attack_type.classes_]

    def score(self, readings) -> list[dict]:
        X = to_frame(readings)
        p_anomaly = self.binary.predict_proba(X)[:, self._anomaly_col]
        type_proba = self.attack_type.predict_proba(X)
        # score_samples: lower = more abnormal, so negate. predict: -1 = outlier.
        novelty_score = -self.novelty.score_samples(X)
        is_novel = self.novelty.predict(X) == -1

        return [
            {
                "p_anomaly": float(p_anomaly[i]),
                "attack_type": self._type_classes[int(np.argmax(type_proba[i]))],
                "type_proba": dict(zip(self._type_classes, map(float, type_proba[i]))),
                "novelty_flag": bool(is_novel[i]),
                "novelty_score": float(novelty_score[i]),
            }
            for i in range(len(X))
        ]

    def score_one(self, reading: dict) -> dict:
        return self.score(reading)[0]


class StreamDetector:
    """Scores one device's readings in order, adding the stateful sensor checks.

    Holds all per-stream settings (threshold, enabled layers), so several streams can share
    one Detector. A reading is an anomaly if any enabled layer flags it; "reasons" says which.
    """

    def __init__(
        self,
        detector: Detector | None = None,
        threshold: float | None = None,
        use_supervised: bool = True,
        use_novelty: bool = True,
        use_sensor_checks: bool = True,
    ):
        self.detector = detector or Detector()
        self.monitor = SensorMonitor(self.detector.metadata["sensor_limits"])
        self.threshold = self.detector.default_threshold if threshold is None else threshold
        self.use_supervised = use_supervised
        self.use_novelty = use_novelty
        self.use_sensor_checks = use_sensor_checks

    def reset(self):
        self.monitor.reset()

    def score(self, reading: dict) -> dict:
        result = self.detector.score_one(reading)
        result["supervised_flag"] = result["p_anomaly"] >= self.threshold
        sensor_reasons = self.monitor.update(reading)
        reasons = []
        if self.use_supervised and result["supervised_flag"]:
            reasons.append(f"model: {result['attack_type']}" if result["attack_type"] != "normal" else "model: anomaly")
        if self.use_novelty and result["novelty_flag"]:
            reasons.append("unlike normal traffic")
        if self.use_sensor_checks:
            reasons.extend(sensor_reasons)
        return {
            **result,
            "sensor_flag": bool(sensor_reasons),
            "anomaly": bool(reasons),
            "reasons": "; ".join(reasons),
        }
