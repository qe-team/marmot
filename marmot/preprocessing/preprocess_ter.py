from __future__ import print_function
import sys
import re
import numpy as np


def parse_hyp_loc_map(line):
    numbers = [int(x) for x in line.split()]
    orig2shifted = {i: j for (j, i) in list(enumerate(numbers))}
    shifted2orig = dict(enumerate(numbers))
    return (orig2shifted, shifted2orig)


def parse_sentence(line_array):
    hyp, ref = [], []
    orig2shifted, shifted2orig = {}, {}
    align, sentence_id = "", ""
    shifts = []
    for line in line_array:
        line_separator = line.find(':')
        line_id = line[:line_separator]
        if line_id == "Hypothesis":
            hyp = [w for w in line[line_separator+2:-1].split()]
        elif line_id == "Reference":
            ref = [w for w in line[line_separator+2:-1].split()]
        elif line_id == "Sentence ID":
            sentence_id = line[line_separator+2:-1]
        elif line_id == "Alignment":
            align = line[line_separator+3:-2]
        elif line_id == "HypLocMap":
            (orig2shifted, shifted2orig) = parse_hyp_loc_map(line[line_separator+2:-1])
        # shift description
        # shift syntax:
        #  [i, j, m/n] (word(s)) -> (word(s))
        #    i -- original position of the first word
        #    j -- original position of the last word
        #    m -- position of the last word in reference
        #    n -- position of the last word in shifted hypothesis
        elif line.startswith('  ['):
            shifts.append(line)
        else:
            continue

    # mapping between original and shifted hypotheses
    mapping_hyp_shift = {i: i for i in range(len(hyp))}
    mapping_shift_hyp = {i: i for i in range(len(hyp))}
    for shift in shifts:
        numbers = [int(n) for n in re.compile('\d+').findall(shift)]
        if len(numbers) < 4:
            print("Bad shift description in the source file", shift)
            continue
        len_shift = numbers[1] + 1 - numbers[0]
        for i in range(len_shift):
            shifted_pos = numbers[3] - len_shift + 1 + i
            mapping_hyp_shift[numbers[0]+i] = shifted_pos
            mapping_shift_hyp[shifted_pos] = numbers[0] + i

    # mapping between reference and hypothesis with shifts
    mapping_ref_shift = {}
    mapping_shift_ref = {}
    ref_cnt, shift_cnt = 0, 0
    for c in align:
        if c == 'D':
            mapping_ref_shift[ref_cnt] = None
            ref_cnt += 1
        elif c == 'I':
            mapping_shift_ref[shift_cnt] = None
            shift_cnt += 1
        else:
            mapping_ref_shift[ref_cnt] = shift_cnt
            mapping_shift_ref[shift_cnt] = ref_cnt
            ref_cnt += 1
            shift_cnt += 1

    # mappings between hypothesis and reference
    mapping_hyp_ref, mapping_ref_hyp = {}, {}
    for i, j in mapping_hyp_shift.items():
        if j in mapping_shift_ref:
            mapping_hyp_ref[i] = mapping_shift_ref[j]
        else:
            mapping_hyp_ref[i] = None
    for i, j in mapping_ref_shift.items():
        if j in mapping_shift_hyp:
            mapping_ref_hyp[i] = mapping_shift_hyp[j]
        else:
            mapping_ref_hyp[i] = None

    hyp = np.array(hyp, dtype=object)
    ref = np.array(ref, dtype=object)
    return (sentence_id, hyp, ref, mapping_hyp_ref, mapping_ref_hyp, align)


def get_features(sentence_id, sentence, labels, good_context):
    good_label = u'GOOD'
    bad_label = u'BAD'
#    print "Sentence: ", sentence
#    print "Labels: ", labels

    assert(len(sentence) == len(labels))

    instances = []

    for i in range(len(labels)):
        prev_word, next_word = "", ""
        good_left, good_right = False, False
        if i == 0:
            prev_word = u"START"
            good_left = True
        else:
            prev_word = sentence[i-1]
        if i+1 == len(labels):
            next_word = u"END"
            good_right = True
        else:
            next_word = sentence[i+1]

        if not good_left:
            good_left = (not good_context or labels[i-1] == 'G')
        if not good_right:
            good_right = (not good_context or labels[i+1] == 'G')

        if good_left and good_right:
            cur_label = good_label if labels[i] == 'G' else bad_label
            instances.append(np.array([sentence_id, i, sentence[i], prev_word, next_word, sentence, cur_label, cur_label]))
    return np.array(instances)


# output format: array of training instances
# each instance is an array of:
#     sentence id, word id, word_i, word_i-1, word_i+1, sentence, label, label
# label appears twice for compatibility with fine-grained error classification
def parse_ter_file(pra_file_name, good_context=True):
    a_file = open(pra_file_name)
    sys.stderr.write("Parse file \'%s\'\n" % pra_file_name)
    features = []
    cur_sentence = []
    for line in a_file:
        cur_sentence.append(line.decode("utf-8"))
        if line.startswith('Score: '):
            # parse once you hit 'Score: '
            (sent_id, hyp, ref, orig2shifted, align) = parse_sentence(cur_sentence)
            if len(hyp) != len(align):
                sys.stderr.write("Hypothesis and alignment map don't match, sentence number %s\n" % sent_id)
                cur_sentence = []
                continue

            err_labels = ""
            for i in range(len(hyp)):
                if align[orig2shifted[i]] == ' ':
                    err_labels += 'G'
                elif align[orig2shifted[i]] == 'S' or align[orig2shifted[i]] == 'I':
                    err_labels += 'B'
            features.extend(get_features(sent_id, hyp, err_labels, good_context))
            features.append([])

            cur_sentence = []
    return np.array(features, dtype=object)


def parse_ter_file_basic(pra_file_name):
    sentences = []
    cur_sentence = []
    for line in open(pra_file_name):
        cur_sentence.append(line.decode("utf-8"))
        # parse once you hit 'Score: '
        if line.startswith('Score: '):
            (sent_id, hyp, ref, mapping_hyp_ref, mapping_ref_hyp, align) = parse_sentence(cur_sentence)
            align_no_insertions = align.translate({ord('I'): None})
            align_no_deletions = align.translate({ord('D'): None})
            if len(hyp) != len(align_no_deletions) or len(ref) != len(align_no_insertions):
                sys.stderr.write("Hypothesis and alignment map don't match, sentence number %s\n" % sent_id)
                cur_sentence = []
                continue

            labels_ref = []
            labels_hyp = []

            labels_map = {i: i for i in ['I','D','S','H']}
            labels_map[' '] = 'OK'
#            labels_map = {i: u'BAD' for i in ['I', 'D', 'S', 'H']}
#            labels_map[' '] = u'OK'
            for i in range(len(ref)):
                cur_char = align_no_insertions[i]
                if cur_char != 'I':
                    labels_ref.append(labels_map[cur_char])
#                elif cur_char == 'S' or cur_char == 'D':
#                    labels_ref.append(u'BAD')
            for i in range(len(hyp)):
                cur_char = ''

                # Chris: changed the following line:
                #if i not in mapping_hyp_ref:
                if i not in mapping_hyp_ref or mapping_hyp_ref[i] is None:
                    cur_char = 'I'
                else:
                    cur_char = align_no_insertions[mapping_hyp_ref[i]]
                if cur_char != 'D':
                    labels_hyp.append(labels_map[cur_char])
 #               elif cur_char == 'S' or cur_char == 'I':
 #                   labels_hyp.append(u'BAD')

            cur_sentence = []
            sentences.append({'hyp': hyp, 'ref': ref, 'labels_hyp': labels_hyp, 'labels_ref': labels_ref})
    return sentences
