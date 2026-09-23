"""Feature definitions and preprocessing shared by training, detection and simulation."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

NUMERIC_FEATURES = [
    "duration",
    "destination_bytes",
    "missed_bytes",
    "dst_ip_bytes",
    "Temperature",
    "Humidity",
]
CATEGORICAL_FEATURES = ["Status"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

BINARY_TARGET = "Label"
TYPE_TARGET = "Type"
ATTACK_TYPES = ["normal", "dos", "mitm"]

# Columns removed from the inputs, and why (see notebooks/01_training_fixed.ipynb).
DROPPED_COLUMNS = {
    "Type": "Is the label written another way (normal <=> Label 0). Now a target, not a feature.",
    "source_ip": "Shortcut: .138 is always normal, .162/.164 are always MITM.",
    "source_port": "Connection identifier, not a device property; attacks use a subset of ports.",
    "timestamp": "Only 3 distinct values, carries no information.",
}


def load_dataset(path) -> pd.DataFrame:
    """Load the raw CSV and remove exact duplicate rows."""
    df = pd.read_csv(path)
    return df.drop_duplicates().reset_index(drop=True)


def to_frame(readings) -> pd.DataFrame:
    """Turn one reading (dict), a list of readings, or a DataFrame into model input."""
    if isinstance(readings, dict):
        readings = [readings]
    df = pd.DataFrame(readings)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Reading is missing features: {missing}")
    return df[FEATURES]


def build_preprocessor(log_duration: bool = False) -> ColumnTransformer:
    """Encode Status and pass numeric features through.

    log_duration compresses the long duration tail (up to ~250k), which matters for
    Isolation Forest because it splits uniformly between a feature's min and max.
    """
    numeric = FunctionTransformer(_log_duration) if log_duration else "passthrough"
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("status", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def _log_duration(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["duration"] = np.log1p(X["duration"])
    return X
