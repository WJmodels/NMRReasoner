# Reproducing the archived runs

See [the detailed Chinese reproduction guide](README.zh-CN.md) for environment reconstruction, every experiment, historical limitations, and cleanup details.

`records/` preserves original commands, environment information, source paths, metrics, training arguments, checks, scripts and reports. `source_snapshot.tar.gz` preserves the code and configuration used before cleanup. `pip-list.json` and `pip-freeze.txt` record the actual environment.

From the project root, recreate the environment described in `runtime.yml`. The historical venv inherited PyTorch/CUDA binaries from an existing environment; creating a fresh Conda environment was only solver-checked, not fully tested. Compare installed versions against the archived environment records.

Preview commands (no execution by default):

```bash
python validation/run_reproduction/replay.py full-2.2m
```

Run into a new output directory:

```bash
python validation/run_reproduction/replay.py full-2.2m --gpu 3 --execute
```

Other profiles: `full-200k`, `smoke`, `training`, `vllm-smoke`, `bidirectional-single`. Small dataset subsets are recreated automatically. Historical Transformers runs explicitly retain that backend. Data and original weights remain in `data/` and `weight/`.

Generated model weights, optimizer states and the disposable venv are removed. This prevents resuming the old optimizer checkpoints; retrain from the original weights instead. Predictions, logs and other large ordinary files are retained as `.gz`; decompress them before using historical paths. `cleanup_manifest.json` records every affected file. Old PID-based monitor/finalizer scripts are audit records only and must not be relaunched.

Exact outputs may differ across hardware and software versions. Earlier failed/superseded training attempts are retained as history; the replay entry point uses the final corrected implementation.
