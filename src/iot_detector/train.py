"""Train the detectors and save them to models/.

Usage:  python -m iot_detector.train
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from . import DATA_PATH, MODEL_DIR
from .features import (
    ATTACK_TYPES,
    BINARY_TARGET,
    DROPPED_COLUMNS,
    FEATURES,
    TYPE_TARGET,
    build_preprocessor,
    load_dataset,
)
from .sensor_checks import learn_limits

RANDOM_STATE = 42
TEST_SIZE = 0.3
# Share of normal training rows Isolation Forest is allowed to flag (its false-alarm budget).
NOVELTY_CONTAMINATION = 0.01
# 100 trees score the same as 300 on this data (binary F1 0.951 vs 0.949) at a third of the
# size and scoring time. Compressed joblib brings all three models to ~1.8 MB.
N_TREES = 100
JOBLIB_COMPRESS = 3


def make_classifier() -> Pipeline:
    return Pipeline(
        [
            ("prep", build_preprocessor()),
            ("model", RandomForestClassifier(n_estimators=N_TREES, class_weight="balanced", random_state=RANDOM_STATE)),
        ]
    )


def make_novelty_detector() -> Pipeline:
    return Pipeline(
        [
            ("prep", build_preprocessor(log_duration=True)),
            ("model", IsolationForest(n_estimators=N_TREES, contamination=NOVELTY_CONTAMINATION, random_state=RANDOM_STATE)),
        ]
    )


def split(df: pd.DataFrame):
    """Stratified split on Type so DoS and MITM both appear in train and test."""
    return train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TYPE_TARGET])


def evaluate(model, X, y, labels) -> dict:
    y_pred = model.predict(X)
    return {
        "report": classification_report(y, y_pred, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, y_pred, labels=labels).tolist(),
        "labels": [str(label) for label in labels],
    }


def feature_importance(pipeline: Pipeline) -> dict:
    names = pipeline.named_steps["prep"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_
    order = np.argsort(importances)[::-1]
    return {str(names[i]).split("__", 1)[-1]: float(importances[i]) for i in order}


def train_all(data_path=DATA_PATH, model_dir=MODEL_DIR, verbose: bool = True) -> dict:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path)
    train_df, test_df = split(df)
    X_train, X_test = train_df[FEATURES], test_df[FEATURES]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    binary = make_classifier().fit(X_train, train_df[BINARY_TARGET])
    attack_type = make_classifier().fit(X_train, train_df[TYPE_TARGET])
    novelty = make_novelty_detector().fit(X_train[train_df[BINARY_TARGET] == 0])

    # Isolation Forest predicts -1 for outliers; map to the Label convention (1 = anomaly).
    novelty_pred = (novelty.predict(X_test) == -1).astype(int)

    metrics = {
        "binary": evaluate(binary, X_test, test_df[BINARY_TARGET], [0, 1]),
        "attack_type": evaluate(attack_type, X_test, test_df[TYPE_TARGET], ATTACK_TYPES),
        "novelty": {
            "report": classification_report(test_df[BINARY_TARGET], novelty_pred, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(test_df[BINARY_TARGET], novelty_pred, labels=[0, 1]).tolist(),
            "labels": ["0", "1"],
        },
        "binary_cv_f1": cross_val_score(make_classifier(), X_train, train_df[BINARY_TARGET], cv=cv, scoring="f1").tolist(),
    }

    joblib.dump(binary, model_dir / "binary_rf.joblib", compress=JOBLIB_COMPRESS)
    joblib.dump(attack_type, model_dir / "attack_type_rf.joblib", compress=JOBLIB_COMPRESS)
    joblib.dump(novelty, model_dir / "novelty_iforest.joblib", compress=JOBLIB_COMPRESS)

    # Reference rows feed the simulator: train normals for the generator, test rows for replay.
    reference = pd.concat([train_df.assign(split="train"), test_df.assign(split="test")]).sort_index()
    reference.to_csv(model_dir / "reference.csv", index_label="row_id")

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": FEATURES,
        "dropped_columns": DROPPED_COLUMNS,
        "attack_types": ATTACK_TYPES,
        "rows": {"after_dedup": len(df), "train": len(train_df), "test": len(test_df)},
        "binary_threshold": 0.5,
        "sensor_limits": learn_limits(train_df[train_df[BINARY_TARGET] == 0].sort_index()),
        "feature_importance": {
            "binary": feature_importance(binary),
            "attack_type": feature_importance(attack_type),
        },
        "metrics": metrics,
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    if verbose:
        print(f"Trained on {len(train_df)} rows, tested on {len(test_df)} (after removing duplicates).")
        for name in ("binary", "attack_type", "novelty"):
            report = metrics[name]["report"]
            print(f"{name:12s} accuracy={report['accuracy']:.4f}  macro F1={report['macro avg']['f1-score']:.4f}")
        print(f"binary 5-fold CV F1: {np.mean(metrics['binary_cv_f1']):.4f}")
        print(f"Saved models to {model_dir}")
    return metadata


if __name__ == "__main__":
    train_all()
