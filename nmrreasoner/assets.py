"""Validate Arrow datasets and locate an unambiguous full checkpoint."""

from pathlib import Path


def load_data(path):
    from datasets import Dataset, concatenate_datasets

    files = sorted(Path(path).glob("*.arrow"))
    if not files:
        raise FileNotFoundError(f"No Arrow shards: {path}")
    data = concatenate_datasets([Dataset.from_file(str(p)) for p in files])
    missing = {"question", "cot", "answer"} - set(data.column_names)
    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")
    if not len(data):
        raise ValueError("Empty dataset")
    return data


def checkpoint(path):
    root = Path(path).resolve()
    candidates = (
        [root]
        if (root / "config.json").is_file()
        else sorted(p.parent for p in root.rglob("config.json"))
    )
    candidates = [
        p
        for p in candidates
        if list(p.glob("*.safetensors")) or list(p.glob("pytorch_model*.bin"))
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one checkpoint under {root}; found {candidates}. Specify the exact checkpoint."
        )
    if not (candidates[0] / "tokenizer_config.json").is_file():
        raise FileNotFoundError(f"Missing tokenizer in {candidates[0]}")
    return candidates[0]


def verify_task(data):
    from spectroscopy import scoring as metrics

    task = metrics.judge_task_from_answer_by_keyword(list(data["answer"]))
    if task not in {"nmr", "nmr_forward"}:
        raise ValueError(f"Task {task!r} is outside this project scope")
    return task
