import unittest
import yaml
import sys,os
import inspect
import numpy as np
import marmot
from marmot.experiment import experiment_utils
import time


def join_with_module_path(loader, node):
    """ define custom tag handler to join paths with the path of the marmot module """
    module_path = os.path.dirname(marmot.__file__)
    resolved = loader.construct_scalar(node)
    return os.path.join(module_path, resolved)

## register the tag handler
yaml.add_constructor('!join', join_with_module_path)

class TestRunExperiment(unittest.TestCase):

    def setUp(self):
        module_path = os.path.dirname(__file__)
        self.module_path = module_path
        test_config = os.path.join(module_path, 'test_data/test_config.yaml')

        with open(test_config, "r") as cfg_file:
            self.config = yaml.load(cfg_file.read())


    def test_import_class(self):
        module_name = [cc['module'] for cc in self.config['context_creators']][0]
        klass = experiment_utils.import_class(module_name)
        self.assertTrue(inspect.isclass(klass))

    def test_import_function(self):
        func_name = 'marmot.preprocessing.parsers.parse_back_translation'
        func = experiment_utils.import_function(func_name)
        self.assertTrue(inspect.isfunction(func))

    def test_call_function(self):
        func_name = 'marmot.preprocessing.parsers.parse_corpus_contexts'
        func = experiment_utils.import_function(func_name)
        interesting_tokens = set(['the','it'])
        corpus_path = os.path.join(self.module_path, 'test_data/corpus.en.1000')
        args = [corpus_path, interesting_tokens]
        result = experiment_utils.call_function(func, args)
        self.assertTrue(len(result) > 0)


    def test_build_context_creator(self):
        testing_cc = self.config['testing']
        context_creator = experiment_utils.build_context_creator(testing_cc)
        self.assertTrue(len(context_creator.get_contexts('and')) > 0)
        self.assertFalse(context_creator.get_contexts('the')[0]['token'] == None)

    def test_build_context_creators(self):
        context_creator_list = self.config['context_creators']
        context_creators = experiment_utils.build_context_creators(context_creator_list)
        self.assertTrue(len(context_creators[0].get_contexts('the')) > 0)

    def test_map_contexts(self):
        context_creator_list = self.config['context_creators']
        context_creators = experiment_utils.build_context_creators(context_creator_list)
        interesting_tokens = set(['the','it'])

        token_contexts = experiment_utils.map_contexts(interesting_tokens, context_creators)
        for token in token_contexts.keys():
            self.assertTrue(token in interesting_tokens)
            self.assertTrue(len(token_contexts[token]) > 0)
            for context in token_contexts[token][:10]:
                self.assertTrue(context['token'] != None)

    def test_build_feature_extractors(self):
        # test construction of feature extractors
        feature_extractor_list = self.config['feature_extractors']
        feature_extractors = experiment_utils.build_feature_extractors(feature_extractor_list)
        from marmot.features.feature_extractor import FeatureExtractor
        for extractor in feature_extractors:
            self.assertTrue(isinstance(extractor, FeatureExtractor))

    def test_map_feature_extractors(self):
        context_creator_list = self.config['context_creators']
        context_creators = experiment_utils.build_context_creators(context_creator_list)
        interesting_tokens = set(['the','it', 'a'])

        token_contexts = experiment_utils.map_contexts(interesting_tokens, context_creators)

        feature_extractor_list = self.config['feature_extractors'][:1]
        feature_extractors = experiment_utils.build_feature_extractors(feature_extractor_list)

        mapped_context = np.hstack([experiment_utils.map_feature_extractors( (token_contexts['the'][0], extractor) ) for extractor in feature_extractors])
        self.assertTrue(isinstance(mapped_context, np.ndarray))
        # uses the TokenCountFeatureExtractor, which returns 3 features
        self.assertTrue(len(mapped_context) == 3)

    def test_contexts_to_features(self):
        context_creator_list = self.config['context_creators']
        context_creators = experiment_utils.build_context_creators(context_creator_list)
        interesting_tokens = set(['the','it', 'a'])

        token_contexts = experiment_utils.map_contexts(interesting_tokens, context_creators)

        feature_extractor_list = self.config['feature_extractors'][:1]
        feature_extractors = experiment_utils.build_feature_extractors(feature_extractor_list)

        workers = 8
        mapped_contexts = experiment_utils.contexts_to_features(token_contexts, feature_extractors, workers=8)

        self.assertEqual(set(mapped_contexts.keys()), set(token_contexts.keys()))
        for tok, feature_vecs in mapped_contexts.items():
            self.assertTrue(feature_vecs.shape[0] == len(token_contexts[tok]))


    def test_contexts_to_features_categorical(self):

        token_contexts = {}
        token_contexts['little'] = [ {'index':1, 'token':u'little', 'target':[u'the', u'little', u'boy'], 'source':[u'le', u'petit', u'garcon'], 'alignments':[[0],[1],[2]], 'source_pos':[u'Art', u'Adj', u'Noun'], 'target_pos':[u'DT', u'JJ', u'NN']}, {'index':1, 'token':u'little', 'target':[u'a', u'little', u'dog'], 'source':[u'un', u'petit', u'chien'], 'alignments':[[0],[1],[2]], 'source_pos':[u'Art', u'Adj', u'Noun'], 'target_pos':[u'DT', u'JJ', u'NN']}, {'index':1, 'token':u'little', 'target':[u'a', u'little', u'cat'], 'source':[u'un', u'petit', u'chat'], 'alignments':[[0],[1],[2]], 'source_pos':[u'Art', u'Adj', u'Noun'], 'target_pos':[u'DT', u'JJN', u'NN']} ]

        feature_extractor_list = self.config['feature_extractors']
        feature_extractors = experiment_utils.build_feature_extractors(feature_extractor_list)

        workers = 8
        mapped_contexts = experiment_utils.contexts_to_features_categorical(token_contexts, feature_extractors, workers=8)

        self.assertEqual(set(mapped_contexts.keys()), set(token_contexts.keys()))
        for tok, feature_vecs in mapped_contexts.items():
            self.assertTrue(len(feature_vecs) == len(token_contexts[tok]))
        context = mapped_contexts['little'][0]
        self.assertEqual( context[0], 3 )
        self.assertEqual( context[1], 3 )
        self.assertAlmostEqual( context[2], 1.0 )
        self.assertEqual( context[3], u'petit' )
        self.assertEqual( context[4], [u'le'] )
        self.assertEqual( context[5], [u'garcon'] )
        self.assertEqual( [ context[6], context[7], context[8], context[9] ], [0,0,0,0] )
        self.assertEqual( context[12], u'JJ' )
        self.assertEqual( context[13], [u'Adj'] )


    def test_binarizers(self):
        contexts = [[3.0, 3.0, 1.0, u'petit', [u'le'], [u'garcon'], 0, 0, 0, 0, 0, 0, u'JJ', [u'Adj']], [3.0, 3.0, 1.0, u'petite', [u'un'], [u'chien'], 0, 0, 0, 0, 2, 0, u'JJ', [u'Adv']], [3.0, 3.0, 1.0, u'petits', [u'un'], [u'chat'], 0, 0, 0, 0, 2, 0, u'JJN', [u'Adj', u'Adv']]]

        binarizers = experiment_utils.fit_binarizers(contexts)
        binarized_features = [ experiment_utils.binarize(val, binarizers) for val in contexts ]
        self.assertTrue( np.allclose(binarized_features[0], np.array([3., 3., 1., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.]) ) )
        self.assertTrue( np.allclose(binarized_features[1], np.array([3., 3., 1., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 2., 0., 0., 0., 1.]) ) )
        self.assertTrue( np.allclose(binarized_features[2], np.array([3., 3., 1., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 2., 0., 1., 1., 1.]) ) )


    def test_time_contexts_to_features(self):
        context_creator_list = self.config['context_creators']
        context_creators = experiment_utils.build_context_creators(context_creator_list)
        interesting_tokens = set(['the','it', 'a'])

        token_contexts = experiment_utils.map_contexts(interesting_tokens, context_creators)

        feature_extractor_list = self.config['feature_extractors'][:1]
        feature_extractors = experiment_utils.build_feature_extractors(feature_extractor_list)

        start = time.time()
        mapped_contexts = experiment_utils.contexts_to_features(token_contexts, feature_extractors)
        finish = time.time() - start
        print "Single: ", finish

        start = time.time()
        mapped_contexts = experiment_utils.contexts_to_features(token_contexts, feature_extractors, workers=10)
        finish = time.time() - start
        print "Multiple: ", finish


    def test_tags_from_contexts(self):
        context_creator_list = self.config['context_creators']
        context_creators = experiment_utils.build_context_creators(context_creator_list)
        interesting_tokens = set(['the','it', 'a'])

        token_contexts = experiment_utils.map_contexts(interesting_tokens, context_creators)

        token_tags = experiment_utils.tags_from_contexts(token_contexts)

        self.assertEqual(set(token_tags.keys()), set(token_contexts.keys()))
        for tok, tag_vector in token_tags.items():
            self.assertTrue(tag_vector.shape[0] == len(token_contexts[tok]))

    def test_filter_contexts(self):
        context_creator_list = self.config['context_creators']
        context_creators = experiment_utils.build_context_creators(context_creator_list)

        fake_token = '_z_z_z'
        interesting_tokens = set(['the','it', 'a', fake_token])
        token_contexts = experiment_utils.map_contexts(interesting_tokens, context_creators)
        self.assertTrue(fake_token in token_contexts)

        filtered_contexts = experiment_utils.filter_contexts(token_contexts, min_total=1)
        self.assertFalse(fake_token in filtered_contexts, 'a token that does not exist should not have len(contexts) >= 1')


    def test_filter_context_class(self):
        context_creator_list = self.config['context_creators2']
        context_creators = experiment_utils.build_context_creators(context_creator_list)

        interesting_tokens = set(['del','pescado'])
        token_contexts = experiment_utils.map_contexts(interesting_tokens, context_creators)
        self.assertTrue('pescado' in token_contexts)

        filtered_contexts = experiment_utils.filter_contexts_class(token_contexts, min_total=self.config['filters']['min_total'], min_class_count=self.config['filters']['min_class_count'], proportion=self.config['filters']['proportion'])
        self.assertTrue('del' in filtered_contexts)
        self.assertFalse('pescado' in filtered_contexts)


    def test_convert_tagset(self):
        wmt_binary_classes = {0 :u'BAD', 1: u'OK'}
        context_creator_list = self.config['context_creators']
        context_creators = experiment_utils.build_context_creators(context_creator_list)

        interesting_tokens = set(['the','it', 'a'])
        token_contexts = experiment_utils.map_contexts(interesting_tokens, context_creators)
        new_token_contexts = experiment_utils.convert_tagset(wmt_binary_classes, token_contexts)
        reverse_map = {v:k for k,v in wmt_binary_classes.items()}
        for idx, tok_and_contexts in enumerate(new_token_contexts.iteritems()):
            tok, contexts = tok_and_contexts
            for idx, context in enumerate(contexts):
                self.assertEqual(reverse_map[context['tag']], token_contexts[tok][idx]['tag'])


# test building and calling through a graph of functions and inputs
import json
from marmot.preprocessing.parsers import parse_corpus_contexts, extract_important_tokens


class TestFunctionTree(unittest.TestCase):

    def setUp(self):
        module_path = os.path.dirname(__file__)
        self.module_path = module_path
        test_config = os.path.join(module_path, 'test_data/sample_function_tree.yaml')
        self.basepath = os.path.dirname(marmot.__file__)
        with open(test_config, "r") as cfg_file:
            self.config = yaml.load(cfg_file.read())

        self.corpus_path = os.path.join(module_path, 'test_data/corpus.en.1000')
        important_tokens_path = os.path.join(self.module_path, 'test_data/training.txt')
        self.interesting_tokens = extract_important_tokens(important_tokens_path)
        self.contexts = parse_corpus_contexts(self.corpus_path, self.interesting_tokens)


    def test_call_sample_function_tree(self):
        graph_obj = self.config['test_obj']
        func = experiment_utils.import_function(graph_obj['func'])
        args = graph_obj['args']
        json_contexts = experiment_utils.function_tree(func, args)

        self.assertTrue(type(json_contexts) == str)
        self.assertListEqual(json.loads(json_contexts), self.contexts)


if __name__ == '__main__':
    unittest.main()
