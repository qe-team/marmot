import unittest
from marmot.evaluation.evaluation_metrics import get_spans, intersect_spans, sequence_correlation


class TestEvaluationUtils(unittest.TestCase):

    def setUp(self):
        self.predictions = []
        cur_pred = []
        for line in open('test_data/hyp'):
            if line.strip() == '':
                self.predictions.append(cur_pred)
                cur_pred = []
            else:
                cur_pred.append(line.strip())
        self.predictions.append(cur_pred)
        
        self.references = []
        cur_ref = []
        for line in open('test_data/ref'):
            if line.strip() == '':
                self.references.append(cur_ref)
                cur_ref = []
            else:
                cur_ref.append(line.strip())
        self.references.append(cur_ref)

    def test_get_spans(self):
        sentence = [1, 1, 0, 1, 0, 1, 1, 1, 0]
        good_s, bad_s = get_spans(sentence)
        # test that right spans are extracted
        self.assertItemsEqual(good_s, [(0, 2), (3, 4), (5, 8)])
        self.assertItemsEqual(bad_s, [(2, 3), (4, 5), (8, 9)])
        all_spans = sorted(good_s + bad_s)
        all_items = [t for a_list in [sentence[b:e] for (b, e) in all_spans] for t in a_list]
        # test that the extracted spans cover the whole sequence
        self.assertItemsEqual(sentence, all_items)

    def test_intersect_spans(self):
        true_sentence = [1, 1, 0, 1, 0, 1, 1, 1, 0, 0]
        sentence = [0, 1, 1, 1, 0, 1, 1, 1, 1, 1]
        good_s, bad_s = get_spans(sentence)
        good_t, bad_t = get_spans(true_sentence)
        res_1 = intersect_spans(good_t, good_s)
        res_0 = intersect_spans(bad_t, bad_s)
        self.assertEqual(res_1, 4)
        self.assertEqual(res_0, 1)

    def test_sequence_correlation(self):
        sent_scores, total = sequence_correlation(self.references, self.predictions, good_label='OK', bad_label='BAD')
        self.assertAlmostEqual(sent_scores[0], 0.316)
        self.assertAlmostEqual(sent_scores[1], 0.8)
        self.assertEqual(total, 0.558)


#    def test_alternative_label(self):
#        sequence_correlation(y_true, y_pred, good_label='OK', bad_label='BAD')

if __name__ == '__main__':
        unittest.main()

