from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATASET = ROOT / "archive"/"yolo"
YAML_FILE = ROOT / "working"/"data.yaml"

WEIGHTS = ROOT / "weights"

MODEL_NAME = "yolo26s.pt"

TRAIN_EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 16

CONFIDENCE = 0.40