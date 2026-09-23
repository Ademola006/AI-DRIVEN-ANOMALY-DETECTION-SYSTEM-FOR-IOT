import pandas as pd
import pytest

from iot_detector import MODEL_DIR


@pytest.fixture(scope="session")
def model_dir():
    if not (MODEL_DIR / "metadata.json").exists():
        pytest.skip("No trained models; run `python -m iot_detector.train` first")
    return MODEL_DIR


@pytest.fixture(scope="session")
def reference(model_dir):
    return pd.read_csv(model_dir / "reference.csv")
