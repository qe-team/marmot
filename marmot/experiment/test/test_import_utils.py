import unittest
import yaml
import os
import inspect
import marmot
from marmot.experiment import import_utils


def join_with_module_path(loader, node):
    """ define custom tag handler to join paths with the path of the marmot module """
    module_path = os.path.dirname(marmot.__file__)
    resolved = loader.construct_scalar(node)
    return os.path.join(module_path, resolved)

# register the tag handler
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
        klass = import_utils.import_class(module_name)
        self.assertTrue(inspect.isclass(klass))

    def test_import_function(self):
        func_name = 'marmot.preprocessing.parsers.parse_back_translation'
        func = import_utils.import_function(func_name)
        self.assertTrue(inspect.isfunction(func))

    def test_call_function(self):
        func_name = 'marmot.preprocessing.parsers.parse_corpus_contexts'
        func = import_utils.import_function(func_name)
        interesting_tokens = set(['the', 'it'])
        corpus_path = os.path.join(self.module_path, 'test_data/corpus.en.1000')
        args = [corpus_path, interesting_tokens]
        result = import_utils.call_function(func, args)
        self.assertTrue(len(result) > 0)


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
        func = import_utils.import_function(graph_obj['func'])
        args = graph_obj['args']
        json_contexts = import_utils.function_tree(func, args)

        self.assertTrue(type(json_contexts) == str)
        self.assertListEqual(json.loads(json_contexts), self.contexts)

if __name__ == '__main__':
    unittest.main()
