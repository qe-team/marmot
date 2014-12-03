# test the evaluation functions
# from evaluate import

# choose_token_subset - get the instances for this token only
# signifigance_test -

import unittest
import os
import inspect
from marmot.evaluation.evaluate import significance_test, get_scores, read_wmt_annotation, evaluate_wmt, evaluate_hashed_predictions


class TestEvaluate(unittest.TestCase):

    def setUp(self):
        module_path = os.path.dirname(__file__)
        gold_standard_path = 'test_data/gold_standard.de-en'
        test_annotations_path = 'test_data/test_annotations.de-en'

        # get sample test data and gold standard
        self.gold_standard_path = os.path.join(module_path, gold_standard_path)
        self.test_annotations_path = os.path.join(module_path, test_annotations_path)

        self.ref_annnotations = read_wmt_annotation(open(self.gold_standard_path))
        self.test_annnotations = read_wmt_annotation(open(self.test_annotations_path))
        self.interesting_tokens = []

    def test_significance_test(self):
        test_options = ['ok', 'bad']
        actual = ['ok', 'bad', 'ok', 'bad', 'ok', 'ok', 'ok', 'ok', 'ok', 'ok']
        hyp = ['ok', 'bad', 'ok', 'bad', 'ok', 'ok', 'ok', 'ok', 'ok', 'ok']
        hyp_res = get_scores(actual, hyp, test_options)

        p_value = significance_test(actual, hyp_res, test_options)
        self.assertTrue(p_value == 0.05)
        p_value = significance_test(actual, hyp_res, test_options, granularity=100)
        self.assertTrue(p_value <= 0.05)

    def test_evaluate_wmt(self):
        evaluation_results = evaluate_wmt(self.ref_annnotations, self.test_annnotations)
        self.assertTrue(evaluation_results['all_data']['p_value'] == 0.05)
        print('WMT evaluation_results')
        print(evaluation_results)

    def test_evaluate_hashed_predictions(self):
        # get sample hashed predictions
        test_ref = {'apple': ['good', 'bad', 'bad'], 'fish': ['good']}
        test_hyp = {'apple': ['good', 'bad', 'bad'], 'fish': ['good']}
        test_labels = set(['good', 'bad'])

        evaluation_results = evaluate_hashed_predictions(test_ref, test_hyp, test_labels)
        print('Token hash evaluation_results')
        print(evaluation_results)
        self.assertTrue(evaluation_results['apple']['weighted_f1'] == 1.0)

if __name__ == '__main__':
    unittest.main()



