"""Bridge to the preserved Unsloth training implementation."""

import os
from pathlib import Path
from .assets import load_data, verify_task


def train(data, model, output, epochs=20, batch_size=64):
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch-size must be positive integers")
    verify_task(load_data(data))
    model = Path(model).expanduser().resolve()
    if not model.is_dir():
        raise FileNotFoundError(f"Local base model directory not found: {model}")
    values = dict(
        REASONER_TRAIN_DATA=str(Path(data).expanduser().resolve()),
        REASONER_BASE_MODEL=str(model),
        REASONER_EPOCHS=str(epochs),
        REASONER_BATCH_SIZE=str(batch_size),
    )
    previous = {key: os.environ.get(key) for key in values}
    cwd = Path.cwd()
    out = Path(output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    try:
        os.environ.update(values)
        os.chdir(out)
        from experiments.fit import main

        return main()
    finally:
        os.chdir(cwd)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
