import os
import sys
from src.config import DEFAULTS  # noqa: E402
from src.genre_classifier.dataset import create_dataset  # noqa: E402


# Add project root to sys.path so `src` is importable when running from scripts/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)



if __name__ == "__main__":
    create_dataset(
        data_dir=DEFAULTS["data_dir"],
        dataset_path=DEFAULTS["dataset_path"],
        segment_length_sec=DEFAULTS.get("segment_length_sec", 10),
    )

