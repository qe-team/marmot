from __future__ import division
import os
import sys
from subprocess import call
from collections import defaultdict
from marmot.features.feature_extractor import FeatureExtractor


class NgramFrequenciesFeatureExtractor(FeatureExtractor):

    def __init__(self, tmp_dir, ngram_count_file=None, corpus=None, srilm=None):
        if srilm is None:
            srilm = os.environ['SRILM'] if 'SRILM' in os.environ else None
        if ngram_count_file is None:
            if corpus is None or not os.path.exists(corpus):
                print("No ngram count file and no corpus provided")
                sys.exit()
            if not os.path.exists(corpus):
                print("Corpus doesn't exist")
                sys.exit()
            ngram_count_binary = os.path.join(srilm, 'ngram-count')
            if srilm is None or not os.path.exists(ngram_count_binary):
                print("No SRILM found, ngram counts can't be extracted")
                sys.exit()
            # TODO: run srilm to get the ngram model
            ngram_count_file = os.path.join(tmp_dir, 'tst_counts')
            call([ngram_count_binary, '-order', '3', '-text', corpus, '-write', ngram_count_file])

        # get ngram counts
        ngrams = defaultdict(list)
        for line in open(ngram_count_file):
            chunks = line[:-1].decode('utf-8').split('\t')
            if len(chunks) != 2:
                print("Wrong format of the ngram file '{}', bad line: {}".format(ngram_count_file, line))
                sys.exit()
            words = chunks[0].split()
            ngrams[len(words)].append((chunks[0], int(chunks[1])))
        self.ngrams = {}
        self.ngram_quartiles = {}
        for order in ngrams:
            sorted_ngrams = sorted(ngrams[order], key=lambda(k, v): v)
            self.ngrams[order] = {i: j for (i, j) in sorted_ngrams}

            ngrams_len = len(sorted_ngrams)
            q1, q2, q3 = int(ngrams_len/4), int(ngrams_len/2), int(ngrams_len*3/4)
            # 1 -- low frequency, 4 -- high frequency
            self.ngram_quartiles[order] = {1: {i: j for (i, j) in sorted_ngrams[:q1]},
                                           2: {i: j for (i, j) in sorted_ngrams[q1:q2]},
                                           3: {i: j for (i, j) in sorted_ngrams[q2:q3]},
                                           4: {i: j for (i, j) in sorted_ngrams[q3:]}}

    def get_quartiles_frequency(self, order, source_token):
        quart_frequencies = []
        ngram_list = [' '.join(source_token[i:i+order]) for i in range(len(source_token) - order + 1)]
        for quart in [1, 2, 3, 4]:
            quart_count = 0
            for ngram in ngram_list:
                if ngram in self.ngram_quartiles[order][quart]:
                    quart_count += 1
            quart_frequencies.append(quart_count/len(ngram_list))
        return quart_frequencies

    def get_features(self, context_obj):
        if len(context_obj['source_token']) == 0:
            return [0 for i in range(15)]

        source_token = context_obj['source_token']
        unigram_quart = self.get_quartiles_frequency(1, source_token)
        bigram_quart = self.get_quartiles_frequency(2, source_token)
        trigram_quart = self.get_quartiles_frequency(3, source_token)

        bigram_list = [' '.join(source_token[i:i+2]) for i in range(len(source_token) - 1)]
        trigram_list = [' '.join(source_token[i:i+3]) for i in range(len(source_token) - 2)]
        percent_unigram = sum([1 for word in source_token if word in self.ngrams[1]])/len(source_token)
        percent_bigram = sum([1 for word in bigram_list if word in self.ngrams[2]])/len(source_token)
        percent_trigram = sum([1 for word in trigram_list if word in self.ngrams[3]])/len(source_token)

        return unigram_quart + bigram_quart + trigram_quart + [percent_unigram, percent_bigram, percent_trigram]

    def get_feature_names(self):
        return ['avg_unigram_quart_1',
                'avg_unigram_quart_2',
                'avg_unigram_quart_3',
                'avg_unigram_quart_4',
                'avg_bigram_quart_1',
                'avg_bigram_quart_2',
                'avg_bigram_quart_3',
                'avg_bigram_quart_4',
                'avg_trigram_quart_1',
                'avg_trigram_quart_2',
                'avg_trigram_quart_3',
                'vg_trigram_quart_4',
                'percent_unigram',
                'percent_bigram',
                'percent_trigram']
