"""IoT anomaly detection: training, detection and an anomaly-injection simulator."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "iot_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"
