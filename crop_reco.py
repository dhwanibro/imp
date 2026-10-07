"""Entry point. The code lives in the `krishi/` package; see README.md and ARCHITECTURE.md.

    python crop_reco.py                                       # baseline (same as the original script)
    python crop_reco.py --weather-model bootstrap --scorer ml_yield
    python crop_reco.py --compare
"""
from krishi.cli import main

if __name__ == "__main__":
    main()
