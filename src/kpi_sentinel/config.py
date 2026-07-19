from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.2

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"