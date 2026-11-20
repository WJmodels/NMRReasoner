"""Regression checks for portability, CLI compatibility and record alignment."""

import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from nmrreasoner.assets import checkpoint, load_data
from nmrreasoner.config import DEFAULT_CONFIG, PROJECT_ROOT, load_config
from nmrreasoner.evaluation import evaluate, vote
from nmrreasoner.records import save_predictions, answer_status
from nmrreasoner.training import train


@contextlib.contextmanager
def working_directory(path):
    previous = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)


class ApplicationChecks(unittest.TestCase):
    def test_preset_paths_do_not_depend_on_working_directory(self):
        expected = Path("/path/to/se_test_arrow")
        with tempfile.TemporaryDirectory() as tmp, working_directory(tmp):
            config = load_config(DEFAULT_CONFIG)
            self.assertEqual(Path(config["data"]), expected)
            self.assertEqual(Path(config["output"]), Path(tmp) / "runs/validation")

    def test_custom_preset_and_cli_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preset = root / "experiment.json"
            preset.write_text(
                json.dumps({"data": "dataset", "model": "weights", "limit": 7})
            )
            with working_directory(root.parent):
                config = load_config(preset, {"data": "override", "limit": 2})
            self.assertEqual(config["data"], str(root.parent / "override"))
            self.assertEqual(config["model"], str(root / "weights"))
            self.assertEqual(config["limit"], 2)

    def test_bad_parameters_are_rejected_before_inference(self):
        for override in [
            {"limit": -1},
            {"beam_width": 0},
            {"top_p": 1.1},
            {"temperature": float("nan")},
            {"max_tokens": 0},
            {"adjust_decimal_places": "false"},
        ]:
            with self.subTest(override=override), self.assertRaises(ValueError):
                load_config(overrides=override)
        with tempfile.TemporaryDirectory() as tmp:
            preset = Path(tmp) / "bad.json"
            for payload in [{"max_token": 5}, {"data": None}, []]:
                preset.write_text(json.dumps(payload))
                with self.assertRaises(ValueError):
                    load_config(preset)

    def test_help_works_without_site_packages(self):
        for entry, args in [
            ("test.py", ["--help"]),
            ("train.py", ["--help"]),
            ("run.py", ["--help"]),
            ("run.py", ["vote", "--help"]),
        ]:
            result = subprocess.run(
                [sys.executable, "-S", str(PROJECT_ROOT / entry), *args],
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout)

    def test_missing_assets_have_concise_cli_error(self):
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "test.py"),
                "inspect",
                "--data",
                "/nonexistent-nmr-data",
            ],
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("错误", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_checkpoint_requires_unambiguous_full_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ["checkpoint-1", "checkpoint-2"]:
                path = root / name
                path.mkdir()
                for filename in [
                    "config.json",
                    "tokenizer_config.json",
                    "model.safetensors",
                ]:
                    (path / filename).touch()
            with self.assertRaises(ValueError):
                checkpoint(root)
            self.assertEqual(checkpoint(root / "checkpoint-1"), root / "checkpoint-1")

    def test_data_shards_are_sorted_and_schema_checked(self):
        from datasets import Dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, question in [("b", "second"), ("a", "first")]:
                directory = root / name
                Dataset.from_dict(
                    {"question": [question], "cot": [""], "answer": ["CCO"]}
                ).save_to_disk(str(directory))
                (directory / "data-00000-of-00001.arrow").rename(root / f"{name}.arrow")
            self.assertEqual(list(load_data(root)["question"]), ["first", "second"])

    def test_generation_status_distinguishes_missing_and_truncated_answers(self):
        for text in ["q<think>unfinished reasoning", "<answer></answer>"]:
            self.assertEqual(answer_status(text), {"has_answer": False, "answer_complete": False})
        self.assertEqual(answer_status("<answer>CCO"), {"has_answer": True, "answer_complete": False})
        self.assertEqual(answer_status("<think>reason</think><answer>CCO</answer>"),
                         {"has_answer": True, "answer_complete": True})

    def test_prediction_formats_preserve_newlines_and_smiles_backslashes(self):
        texts = [
            ["q\n<think>reason</think><answer>C/C=C\\C</answer>", "second candidate"]
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_predictions(root, texts)
            self.assertEqual(
                json.loads((root / "predictions.jsonl").read_text())["text"], texts[0]
            )
            self.assertEqual(
                (root / "predictions.txt").read_text().splitlines(),
                [texts[0][0].replace("\n", "\\n")],
            )

    def test_evaluate_and_vote_reject_misaligned_predictions(self):
        data = {"answer": ["CCO", "CCO"]}
        # Use a real two-row dataset to exercise public service validation.
        from datasets import Dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Dataset.from_dict(
                {**data, "question": ["q", "q"], "cot": ["", ""]}
            ).save_to_disk(str(root / "data"))
            predictions = root / "predictions.txt"
            predictions.write_text("<answer>CCO</answer>\n")
            with self.assertRaisesRegex(ValueError, "count"):
                evaluate(root / "data", predictions, root / "metrics.json")
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                json.dumps({"text": ["<answer>CCO</answer>"]}) + "\n"
            )
            with self.assertRaisesRegex(ValueError, "count"):
                vote(root / "data", [predictions], root / "vote")
            self.assertFalse((root / "vote").exists())

    def test_training_restores_directory_and_environment_on_failure(self):
        from experiments import fit

        cwd = Path.cwd()
        before = os.environ.get("REASONER_EPOCHS")
        with tempfile.TemporaryDirectory() as tmp, patch(
            "nmrreasoner.training.load_data", return_value={"answer": ["CCO"]}
        ), patch.object(fit, "main", side_effect=RuntimeError("training failed")):
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                train(tmp, tmp, Path(tmp) / "output", epochs=2)
        self.assertEqual(Path.cwd(), cwd)
        self.assertEqual(os.environ.get("REASONER_EPOCHS"), before)


if __name__ == "__main__":
    unittest.main()
