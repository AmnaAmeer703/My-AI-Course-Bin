from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATASET = ROOT / "archive"/"css-data"
YAML_FILE = ROOT / "working"/"data.yaml"

WEIGHTS = ROOT / "weights"

MODEL_NAME = "yolo11s.pt"

TRAIN_EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 16

CONFIDENCE = 0.40