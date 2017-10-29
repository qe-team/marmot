import sys
from marmot.features.feature_extractor import FeatureExtractor
from gensim.corpora import TextCorpus


class OOVFeatureExtractor(FeatureExtractor):
    '''
    Feature that indicates presence of OOV words in the source phrase.
    Values:
       0 -- no OOV words
       1 -- 1 or more OOV words
    '''

    def __init__(self, corpus_file):
        corpus = TextCorpus(input=corpus_file)
        self.words = corpus.dictionary.values()

    def get_features(self, context_obj):
        #sys.stderr.write("Start OOVFeatureExtractor\n")
        # no source -- no OOVs
        if 'source_token' not in context_obj or len(context_obj['source_token']) == 0:
            #sys.stderr.write("Finish OOVFeatureExtractor\n")
            return ['0']

        for word in context_obj['source_token']:
            if word not in self.words:
                #sys.stderr.write("Finish OOVFeatureExtractor\n")
                return ['1']

        #sys.stderr.write("Finish OOVFeatureExtractor\n")
        return ['0']

    def get_feature_names(self):
        return ['OOV_words']
