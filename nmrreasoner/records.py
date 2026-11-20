"""Persist predictions in the original JSONL/TXT formats."""

import json
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def save_predictions(out, texts):
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for candidates in texts:
            handle.write(json.dumps({"text": candidates}, ensure_ascii=False) + "\n")
    with (out / "predictions.txt").open("w", encoding="utf-8") as handle:
        for candidates in texts:
            handle.write(candidates[0].replace("\n", "\\n").replace("\r", "\\r") + "\n")


def answer_status(text):
    """Distinguish missing/truncated answers from a completed answer section."""
    if "<answer>" not in text:
        return {"has_answer": False, "answer_complete": False}
    content = text.rsplit("<answer>", 1)[1]
    answer = content.split("</answer>", 1)[0].strip()
    return {
        "has_answer": bool(answer),
        "answer_complete": bool(answer) and "</answer>" in content,
    }
