"""Compatibility exports for existing scripts and Python callers."""

from nmrreasoner.assets import load_data, checkpoint, verify_task
from nmrreasoner.config import PROJECT_ROOT
from nmrreasoner.inference import infer
from nmrreasoner.cli import main

if __name__ == "__main__":
    main()
