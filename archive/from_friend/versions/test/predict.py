"""Compatibility entry point for the reference/graph blended predictor."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gcn_predict import main


if __name__ == '__main__':
    main()
