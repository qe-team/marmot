import unittest
import yaml
import os
import marmot
from marmot.experiment import experiment_utils
from marmot.experiment import learning_utils


## define custom tag handler to join paths with the path of the marmot module
def join_with_module_path(loader, node):
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

    def test_token_classifiers(self):
        interesting_tokens = set(['the','it', 'a'])
        context_creators = experiment_utils.build_context_creators(self.config['context_creators'])
        token_contexts = experiment_utils.map_contexts(interesting_tokens, context_creators)

        feature_extractors = experiment_utils.build_feature_extractors(self.config['feature_extractors'])
        token_context_features = experiment_utils.token_contexts_to_features(token_contexts, feature_extractors)
        binarizers = experiment_utils.fit_binarizers(experiment_utils.flatten(token_context_features.values()))
        token_context_features = {k: [experiment_utils.binarize(v, binarizers) for v in val] for k, val in token_context_features.items()}

        token_context_tags = experiment_utils.tags_from_contexts(token_contexts)

        # train the classifier for each token
        classifier_type = experiment_utils.import_class(self.config['learning']['classifier']['module'])

        classifier_map = learning_utils.token_classifiers(token_context_features, token_context_tags, classifier_type)
        self.assertEqual(set(token_context_tags.keys()), set(classifier_map.keys()))
        for tok, classifier in classifier_map.items():
            self.assertTrue(hasattr(classifier, 'predict'))


if __name__ == '__main__':
    unittest.main()
