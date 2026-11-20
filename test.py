"""Inference and evaluation entry point.

Examples:
    python test.py --config experiments/local.json --data /path/to/test_arrow --model /path/to/model
    python test.py --data /path/to/test_arrow --model /path/to/model --limit 0 --output runs/full
    python test.py inspect --data /path/to/test_arrow --model /path/to/model
    python test.py evaluate --data DATA --predictions PREDICTIONS.txt --output metrics.json
    python test.py vote --data DATA --predictions first.jsonl second.jsonl --output runs/vote
"""
import sys

from workflows import main as run_workflow


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in {"infer", "inspect", "evaluate", "vote"}:
        args.insert(0, "infer")
    return run_workflow(args)


if __name__ == "__main__":
    main()
