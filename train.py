"""Supervised fine-tuning entry point.

Example:
    python train.py --data /path/to/train_data --model /path/to/base_model \
        --epochs 20 --batch-size 64 --output runs/training
"""
import sys

from workflows import main as run_workflow


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    return run_workflow(["train", *args])


if __name__ == "__main__":
    main()
