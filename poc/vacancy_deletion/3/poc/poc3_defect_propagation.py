"""Wrapper entrypoint for vacancy_deletion POC 3."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from defect_propagation_972 import main

if __name__ == "__main__":
    main()
