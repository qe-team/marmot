from marmot.features.feature_extractor import FeatureExtractor


class NEFeatureExtractor(FeatureExtractor):
    '''
    Presence/absence of named entities in source and target phrases.
    Named entity = word with 1st capital letter
    '''

    def get_features(self, context_obj):
        src_ne, tg_ne = 0, 0
        for word in context_obj['token']:
            if word[0].isupper():
                tg_ne = 1
        for word in context_obj['source_token']:
            if word[0].isupper():
                src_ne = 1
        return [src_ne, tg_ne]

    def get_feature_names(self):
        return ['named_entity_source', 'named_entity_target']
