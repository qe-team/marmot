from __future__ import print_function, division
from sklearn.metrics import f1_score



def write_res_to_file(test_file, test_predictions, output_file=''):
    if output_file == '':
        output_file = test_file+'.predictions'

    output = open(output_file, 'w')
    try:
        for idx, line in enumerate(open(test_file)):
            chunks = line.decode('utf-8').strip().split('\t')
            prefix = u'\t'.join(chunks[:5])
            # TODO: check if number of strings and predictions match
            output.write('%s\t%s\n' % (prefix.encode('utf-8'), test_predictions[idx].encode('utf-8')))
    except IndexError:
        print('Predictions size:', len(test_predictions), ', current number: ', idx)
    finally:
        output.close()

    return output_file


def compare_vocabulary(datasets):
    '''
    :param datasets: a list of datasets, which are all lists of token sequences
    :return: a list of objects describing each dataset
    '''

    def get_vocab(dataset):
        return set([w for seq in dataset for w in seq])

    vocabs = [get_vocab(dataset) for dataset in datasets]
    common_vocab = set.intersection(*vocabs)
    out = []
    for i, vocab in enumerate(vocabs):
        out.append({'coverage': len(common_vocab) / len(vocab)})

    return out


# evaluation without checking the sentence numbers
# odd_col -- number of columns that should be ignored (e.g. system ID)
def evaluate_simple(ref_file, hyp_file, odd_col=0, check_words=True, average='weighted'):
    tags_ref, tags_hyp = [], []
    tags_dict = {u'BAD': 0, u'OK': 1}
    for idx, (ref, hyp) in enumerate(zip(open(ref_file), open(hyp_file))):
        chunks_ref = ref.decode('utf-8').strip().split('\t')
        chunks_hyp = hyp.decode('utf-8').strip().split('\t')
        if chunks_ref[2] != chunks_hyp[2+odd_col] and check_words:
            print("Words don't match at string", idx)
            return -1
        tags_ref.append(chunks_ref[-1])
        tags_hyp.append(chunks_hyp[-1])
#        all_tags.append(chunks_ref[-1])
#        all_tags.append(chunks_hyp[-1])

#    return f1_score([tags_dict[i] for i in tags_ref], [tags_dict[i] for i in tags_hyp])
    return f1_score([tags_dict[i] for i in tags_ref], [tags_dict[i] for i in tags_hyp], average=average)
