#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, codecs
import numpy as np

issues_accuracy = ['Terminology', 'Mistranslation', 'Omission', 'Addition', 'Untranslated', 'Accuracy']
issues_fluency = ['Agreement', 'Capitalization', 'Fluency', 'Function_words',
                         'Grammar', 'Morphology_(word_form)', 'Style/register',
                         'Typography', 'Unintelligible', 'Word_order',
                         'Tense/aspect/mood', 'Punctuation', 'Spelling',
                         'Part_of_speech']

def read_annotation(f):
    anno = {}
    for line_number, line in enumerate(f):
        line = line.strip().split()
        assert len(line) == 6, "line %d: expected 6 elements per line but found %d\n" %(line_number, len(line))
        sid, wid, w, a1, a2, a3 = line
        sid = sid.split('.')
        assert len(sid) == 2 and sid[0].isdigit() and sid[1].isdigit(), \
                "line %d: first entry (sentence id) must be if format X.Y\n" %line_number
        assert wid.isdigit(), "line %d: second entry (word index) must be integer\n" %line_number
        sid = (int(sid[0]), int(sid[1]))
        wid = int(wid)

        assert a1.lower() == "ok" or \
               a1 in issues_accuracy or \
               a1.lower() in map(str.lower, issues_accuracy) or \
               a1 in issues_fluency or \
               a1.lower() in map(str.lower, issues_fluency), \
                "line %d: unexpected error category %s\n" %(line_number, a1)
        assert a2.lower() in ['ok', 'fluency', 'accuracy'], "line %d: unexpected error category %s\n" %(line_number, a2)
        assert a3.lower() in ['ok', 'bad'], "line %d: unexpected error category %s\n" %(line_number, a3)

        if not sid in anno:
            anno[sid] = {}
        assert not wid in anno[sid], "line %d: duplicate entry for s%d:w%d" %(line_number, sid, wid)
        anno[sid][wid] = [a1.lower(), a2.lower(), a3.lower(), w]
    return anno

def compare_anno(anno1, anno2):
    for sid in anno1:
        #print(sid)
        assert sid in anno2, "s%d only found in one file\n" %(sid)
        for wid in anno1[sid]:
            assert wid in anno2[sid], "s%d:w%d only found in one file\n" %(sid, wid)




def get_precision(tp, fp):
    if tp > 0:
        return float(tp)/(tp+fp)
    return 0.

def get_recall(tp, fn):
    if tp > 0:
        return float(tp)/(tp+fn)
    return 0.

def get_f1(tp, fn, fp):
    precision = get_precision(tp, fp)
    recall = get_recall(tp, fn)
    f1 = 0.
    if precision * recall > 0:
        f1 = 2. * precision*recall/(precision+recall)
    return f1

def matrix(n):
    return [[0]*n for i in range(n)]

def eval_sub_mute(anno1, anno2, idx, options, name):
    options = map(str.lower, options)
    short_options = [o[:7] for o in options]
    cf = matrix(len(options))
    for sid in anno1:
        for wid in anno1[sid]:
            r = anno1[sid][wid][idx]
            r = options.index(r)
            p = anno2[sid][wid][idx]
            p = options.index(p)
            cf[p][r] += 1

    weighted_average_f1 = 0.
    norm = 0
    for i in range(len(options)):
        tn, tp, fn, fp = 0.,0.,0.,0.
        tp = cf[i][i]
        fp = sum(cf[i]) - tp
        fn = sum(l[i] for l in cf) - tp
        f1 = get_f1(tp, fn, fp)
        if i != options.index('ok'):
            weighted_average_f1 += f1 * (tp + fn)
            norm += tp + fn
    return weighted_average_f1/norm


def eval_sub(anno1, anno2, idx, options, name):
    options = map(str.lower, options)
    short_options = [o[:7] for o in options]
    cf = matrix(len(options))
    for sid in anno1:
        for wid in anno1[sid]:
            r = anno1[sid][wid][idx]
            r = options.index(r)
            p = anno2[sid][wid][idx]
            p = options.index(p)
            cf[p][r] += 1

    print "----- Results for %s: -----" %name
    print "-------------------------------------"
    print "\tREFERENCE"
    print "PREDICT\t", "\t".join(short_options)
    for linenr, line in enumerate(cf):
        print "%s\t" %short_options[linenr],
        print "\t".join(map(str,line))
    print "-------------------------------------"
    weighted_average_f1 = 0.
    norm = 0
    for i in range(len(options)):
        print('i is: ' + str(i))
        tn, tp, fn, fp = 0.,0.,0.,0.
        tp = cf[i][i]
        fp = sum(cf[i]) - tp
        fn = sum(l[i] for l in cf) - tp
        f1 = get_f1(tp, fn, fp)
# Chris: this line is NOT the weighted average for the binary task
        if i != options.index('ok'):
            weighted_average_f1 += f1 * (tp + fn)
            norm += tp + fn
        print "F1 %24s: %f" %(options[i], f1)
    print "\n   %24s: %f" %("WEIGHTED AVG", weighted_average_f1/norm)
    print "-------------------------------------"
    return weighted_average_f1/norm

# Chris - working - eval only by the words we care about
def eval_submission_subset(anno1, anno2, idx, options, name, token_set):
    options = map(str.lower, options)
    short_options = [o[:7] for o in options]
    cf = matrix(len(options))

    for sid in anno1:
        for wid in anno1[sid]:
            word = anno1[sid][wid][-1].decode('utf8')
            #print('checking: ' + word)
            if word[:word.find('_')] in token_set:
                #print(anno1[sid][wid][-1] + ' is in token_set')
                r = anno1[sid][wid][idx]
                r = options.index(r)
                p = anno2[sid][wid][idx]
                p = options.index(p)
                cf[p][r] += 1

    print "----- Results for %s: -----" %name
    print "-------------------------------------"
    print "\tREFERENCE"
    print "PREDICT\t", "\t".join(short_options)
    for linenr, line in enumerate(cf):
        print "%s\t" %short_options[linenr],
        print "\t".join(map(str,line))
    print "-------------------------------------"
    weighted_average_f1 = 0.
    norm = 0
    for i in range(len(options)):
        tn, tp, fn, fp = 0.,0.,0.,0.
        tp = cf[i][i]
        fp = sum(cf[i]) - tp
        fn = sum(l[i] for l in cf) - tp
        f1 = get_f1(tp, fn, fp)
# Chris - for the binary case, this only gives f1 for the negative class
#        if i != options.index('ok'):
        weighted_average_f1 += f1 * (tp + fn)
        norm += tp + fn
        print "F1 %24s: %f" %(options[i], f1)
    print "\n   %24s: %f" %("WEIGHTED AVG", weighted_average_f1/norm)
    print "-------------------------------------"
    return weighted_average_f1/norm


def eval_submission_subset_mute(anno1, anno2, idx, options, name, token_set):
    options = map(str.lower, options)
    short_options = [o[:7] for o in options]
    cf = matrix(len(options))

    for sid in anno1:
        for wid in anno1[sid]:
            word = anno1[sid][wid][-1].decode('utf8')
            if word[:word.find('_')] in token_set:
                r = anno1[sid][wid][idx]
                r = options.index(r)
                p = anno2[sid][wid][idx]
                p = options.index(p)
                cf[p][r] += 1

    weighted_average_f1 = 0.
    norm = 0
    for i in range(len(options)):
        tn, tp, fn, fp = 0.,0.,0.,0.
        tp = cf[i][i]
        fp = sum(cf[i]) - tp
        fn = sum(l[i] for l in cf) - tp
        f1 = get_f1(tp, fn, fp)
# Chris - for the binary case, this only gives f1 for the negative class
#        if i != options.index('ok'):
        weighted_average_f1 += f1 * (tp + fn)
        norm += tp + fn
    return weighted_average_f1/norm


def eval_a1(anno1, anno2):
    options = ["ok"] + issues_fluency + issues_accuracy
    eval_sub(anno1, anno2, 0, options, "multiclass")

def eval_a2(anno1, anno2):
    options = ['ok', 'fluency', 'accuracy']
    #OPTIONS = map(str.lower, issues_fluency + issues_accuracy + ["OK"])
    eval_sub(anno1, anno2, 1, options, "3-class")

def eval_a3_subset(anno1, anno2, token_subset):
    options = ['ok', 'bad']
    relevant_toks = set(codecs.open(token_subset, encoding='utf8').read().split('\n'))
    print(relevant_toks)
    eval_submission_subset(anno1, anno2, 2, options, "binary", relevant_toks)

def eval_a3(anno1, anno2):
    options = ['ok', 'bad']
    eval_sub(anno1, anno2, 2, options, "binary")

def eval_a3_significance(anno1, anno2):
    options = ['ok', 'bad']
    res = eval_sub(anno1, anno2, 2, options, "binary")
    rand_res = []
    for i in range(20):
        rand_anno = generate_random(anno1, 2, options)
        rand_res.append( eval_sub(anno1, rand_anno, 2, options, 'binary') )
    print rand_res
    if all( [max(0,res-i) for i in rand_res] ):
        print "Statistically significant with p = 0.05"
    else:
        print "The result is not statistically significant"
   

def eval_a3_subset_significance(anno1, anno2, token_subset):
    options = ['ok', 'bad']
    relevant_toks = set(codecs.open(token_subset, encoding='utf8').read().split('\n'))
    print(relevant_toks)
    res = eval_submission_subset(anno1, anno2, 2, options, "binary", relevant_toks)
    rand_res = []
    for i in range(20):
        rand_anno = generate_random(anno1, 2, options)
        rand_res.append( eval_submission_subset_mute(anno1, rand_anno, 2, options, 'binary', relevant_toks) )
    print rand_res
    if all( [max(0,res-i) for i in rand_res] ):
        print "Statistically significant with p = 0.05"
    else:
        print "The result is not statistically significant"
 

#Varvara - generate random annotations for <idx> column
#Is it better to generate annotations separately for every type, or to generate only one fine-grained?
def generate_random(anno, idx, options):
    rand_anno = {}
    probs = np.zeros(len(options))
    for sid in anno:
        for wid in anno[sid]:
            cur_val = options.index( anno[sid][wid][idx] )
            probs[cur_val] += 1.0

    #observed distribution
    probs = probs/sum([len(anno[i]) for i in anno])
    for sid in anno:
        if sid not in rand_anno:
            rand_anno[sid] = {}
        for wid in anno[sid]:
            rand_anno[sid][wid] = anno[sid][wid][:idx]+[options[np.random.multinomial(1,probs).argmax()]]+anno[sid][wid][idx+1:]

    return rand_anno
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('ref', help="reference annotations")
    parser.add_argument('sub', help="submission annotations")
    parser.add_argument('--token_subset', help="subset of tokens to evaluate")
    args = parser.parse_args(sys.argv[1:])

    ref = read_annotation(open(args.ref))
    submission = read_annotation(open(args.sub))

    token_subset = ""
    if args.token_subset:
        token_subset = args.token_subset

    compare_anno(ref, submission)
    compare_anno(submission, ref)

    # Chris: other evaluation types commented out, since we currently only care about GOOD / BAD
    # eval_a1(ref, submission)
    # eval_a2(ref, submission)

    if token_subset:
        eval_a3_subset_significance(ref, submission, token_subset)
    else:
        eval_a3_significance(ref, submission)

