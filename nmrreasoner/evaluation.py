"""Evaluation and voting services, with input-alignment checks."""

import json
from pathlib import Path
from .assets import load_data, verify_task
from .records import write_json


def evaluate(data_path, predictions, output):
    from spectroscopy.scoring import test_model_predict_result

    data = load_data(data_path)
    verify_task(data)
    count = len(Path(predictions).read_text(encoding="utf-8").splitlines())
    if count != len(data):
        raise ValueError(
            f"Prediction count {count} != dataset count {len(data)}. Use the run's ground_truth directory."
        )
    result = test_model_predict_result(str(data_path), str(predictions))
    write_json(output, result)
    print(f"Metrics: {Path(output).resolve()}")
    return result


def vote(data_path, predictions, output, topk=10, workers=1):
    from spectroscopy.voting import (
        cot_sc_multi_jsonl_topk_result_to_dict_key_str_topk_value_float_acc,
    )

    data = load_data(data_path)
    if verify_task(data) != "nmr":
        raise ValueError(
            "Voting requires SMILES structure predictions (nmr); use evaluate for nmr_forward chemical shifts."
        )
    if topk < 1 or workers < 1:
        raise ValueError("topk and workers must be positive integers")
    for path in predictions:
        count = 0
        with Path(path).open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ValueError(
                        f"{path}:{number}: empty row would break sample alignment"
                    )
                obj = json.loads(line)
                if not isinstance(obj, dict) or not isinstance(obj.get("text"), list):
                    raise ValueError(
                        f'{path}:{number}: expected {{"text": ["candidate", ...]}}'
                    )
                count += 1
        if count != len(data):
            raise ValueError(
                f"{path}: prediction count {count} != dataset count {len(data)}"
            )
    return cot_sc_multi_jsonl_topk_result_to_dict_key_str_topk_value_float_acc(
        str(data_path),
        [str(p) for p in predictions],
        str(output),
        "consensus",
        topk=topk,
        num_process=workers,
        check_nmr_formula=True,
    )
