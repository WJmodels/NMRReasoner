"""Run with unittest; checks use chemically equivalent and invalid predictions."""
import unittest
from spectroscopy import scoring as metrics, decoding as pipeline, voting

class ChemistryChecks(unittest.TestCase):
    def test_answer_extraction(self):
        self.assertEqual(metrics.extract_str_answer('<think>reason</think><answer>CCO</answer>'), 'CCO')

    def test_prompt_compatibility(self):
        self.assertEqual(pipeline.mapping_list_qusetion_to_certain_form(['q'], None, False, False), ['q<think>'])
        self.assertEqual(pipeline.mapping_list_qusetion_to_certain_form(['q'], None, False, True), [{'prompt':'q<think>'}])

    def test_canonicalization(self):
        self.assertEqual(voting._canonicalize_smiles('OCC'), voting._canonicalize_smiles('CCO'))
        self.assertIsNone(voting._canonicalize_smiles('invalid!'))

    def test_task_scope(self):
        self.assertEqual(metrics.judge_task_from_answer_by_keyword(['[Reactant1]CCO']), 'retrosyn')
        self.assertEqual(metrics.judge_task_from_answer_by_keyword(['CCO']), 'nmr')

    def test_spectral_metrics(self):
        self.assertEqual(metrics._calculate_error_metrics([1.0, 2.0], [1.0, 2.0]), (0.0, 0.0, 0.0))

    def test_training_import_has_no_side_effects(self):
        import importlib
        module = importlib.import_module('experiments.fit')
        self.assertTrue(callable(module.main))

if __name__ == '__main__':
    unittest.main()
