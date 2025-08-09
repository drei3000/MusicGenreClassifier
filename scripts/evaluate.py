import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import DEFAULTS  # noqa: E402
from src.genre_classifier.model import evaluate_on_test_set  # noqa: E402

if __name__ == "__main__":
    evaluate_on_test_set(
        model_path=DEFAULTS["model_path"],
        dataset_path=DEFAULTS["dataset_path"],
        level="file",
    )

