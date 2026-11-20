"""Configuration precedence: defaults < JSON preset < explicit CLI options.

Preset data/model paths are relative to that JSON file. CLI paths and output
paths are relative to the caller's working directory. Absolute paths still work.
"""

import json
import math
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "experiments" / "local.json"
DEFAULTS = {
    "data": "/path/to/se_test_arrow",
    "model": "/path/to/se_model",
    "output": "runs/inference",
    "limit": 3,
    "max_tokens": 5000,
    "backend": "vllm",
    "device": "cuda:0",
    "beam_width": 1,
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 20,
    "adjust_decimal_places": False,
    "c_shift_decimal_place": 1,
    "h_shift_decimal_place": 2,
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.7,
}


def resolve_path(value, base):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Paths must be non-empty strings")
    path = Path(os.path.expandvars(value)).expanduser()
    return str((base / path).resolve() if not path.is_absolute() else path.resolve())


def load_config(path=None, overrides=None):
    config = dict(DEFAULTS)
    if path is not None:
        path = Path(path).expanduser().resolve()
        preset = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(preset, dict):
            raise ValueError("Configuration must be a JSON object")
        unknown = set(preset) - set(DEFAULTS)
        if unknown:
            raise ValueError(
                f'Unknown configuration keys: {", ".join(sorted(unknown))}'
            )
        for key in ("data", "model"):
            if key in preset:
                preset[key] = resolve_path(preset[key], path.parent)
        config.update(preset)
    for key, value in (overrides or {}).items():
        if key in DEFAULTS and value is not None:
            config[key] = value
    for key in ("data", "model", "output"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be a non-empty path")
        config[key] = resolve_path(config[key], Path.cwd())
    validate_config(config)
    return config


def validate_config(config):
    for key, minimum in [
        ("limit", 0),
        ("max_tokens", 1),
        ("beam_width", 1),
        ("c_shift_decimal_place", 0),
        ("h_shift_decimal_place", 0),
        ("tensor_parallel_size", 1),
    ]:
        value = config.get(key, DEFAULTS[key])
        if type(value) is not int or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    top_k = config.get("top_k", 20)
    if type(top_k) is not int or top_k < -1:
        raise ValueError("top_k must be an integer >= -1")
    for key, low, high in [
        ("temperature", 0, None),
        ("top_p", 0, 1),
        ("gpu_memory_utilization", 0, 1),
    ]:
        value = config.get(key, DEFAULTS[key])
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{key} must be a finite number")
        if (
            value < low
            or (key != "temperature" and value == 0)
            or (high is not None and value > high)
        ):
            raise ValueError(f"Invalid {key}: {value}")
    if config.get("backend", "vllm") not in ("transformers", "vllm"):
        raise ValueError("backend must be transformers or vllm")
    if not isinstance(config.get("device", "cuda:0"), str):
        raise ValueError("device must be a string, such as cuda:0 or cpu")
    if type(config.get("adjust_decimal_places", False)) is not bool:
        raise ValueError("adjust_decimal_places must be true or false")
