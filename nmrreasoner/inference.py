"""Dataset → prompts → generation → original chemical scorer → run record."""

import time
from pathlib import Path
from .assets import load_data, checkpoint, verify_task
from .backends import generate
from .records import save_predictions, write_json


def infer(config):
    import torch
    from transformers import AutoTokenizer
    from spectroscopy import decoding as pipeline, scoring as metrics
    from .config import validate_config

    validate_config(config)
    started = time.time()
    data = load_data(config["data"])
    total = len(data)
    task = verify_task(data)
    limit = config.get("limit", 3)
    if limit < 0:
        raise ValueError("limit must be >= 0 (0 means full dataset)")
    if limit:
        data = data.select(range(min(limit, len(data))))
    model_dir = checkpoint(config["model"])
    out = Path(config.get("output", "runs/inference")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    data.save_to_disk(str(out / "ground_truth"))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    questions = [
        pipeline.standardize_shift_decimal_place(
            q,
            config.get("adjust_decimal_places", False),
            config.get("c_shift_decimal_place", 1),
            config.get("h_shift_decimal_place", 2),
        )
        for q in data["question"]
    ]
    prompts = pipeline.mapping_list_qusetion_to_certain_form(
        questions, tokenizer, pipeline.detect_qwen_model_type(str(model_dir)), False
    )
    backend = config.get("backend", "vllm")
    print(
        f"Task: {task} | Samples: {len(data)}/{total} | Backend: {backend}", flush=True
    )
    print(f"Checkpoint: {model_dir}\nOutput: {out}", flush=True)
    texts, details = generate(config, model_dir, tokenizer, prompts)
    save_predictions(out, texts)
    result = metrics.test_model_predict_result(
        str(out / "ground_truth"), str(out / "predictions.txt")
    )
    write_json(out / "metrics.json", result)
    report = {
        "project": "NMRReasoner",
        "task": task,
        "model": str(model_dir),
        "data": str(Path(config["data"]).resolve()),
        "dataset_rows": total,
        "evaluated_rows": len(data),
        "backend": backend,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "config": config,
        "elapsed_seconds": time.time() - started,
        "generation": details,
        "metrics": result,
    }
    write_json(out / "validation.json", report)
    print(f"Results: {out}", flush=True)
    return report
