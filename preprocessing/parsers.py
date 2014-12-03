#!/usr/bin/python
# -*- coding: utf-8 -*-

# A parser takes some input, and returns a list of contexts in the format:  { 'token': <token>, index: <idx>, 'source': [<source toks>]', 'target': [<target toks>], 'tag': <tag>}
# return a context object from an iterable of contexts, and a set of interesting tokens
from word_level.util.simple_corpus import SimpleCorpus
from collections import defaultdict
from subprocess import Popen, call
from nltk import word_tokenize
from word_level.util.force_align import Aligner
import time
import os, sys


def extract_important_tokens(corpus_file, min_count=1):
    corpus = SimpleCorpus(corpus_file)
    word_counts = defaultdict(int)
    for context in corpus.get_texts():
        for word in context:
            word_counts[word] += 1
    return set([k for k,v in word_counts.items() if v >= min_count])


def create_new_instance(token=None, idx=None, source=None, target=None, label=None):
    return {'token': token, 'index': idx, 'source': source, 'target': target, 'tag': label}

# create context object with any number of fields
# elements - values of fields
# elem_labels - names of fields
def create_new_instance_additional(token, idx, target=None, label=None, elements=[], elem_labels=[]):
    context_obj = {'token': token, 'index': idx, 'target': target, 'tag': label}
    context_obj.update( {l:el for l,el in zip(elem_labels, elements)} )
    return context_obj


# by default, this function returns postive contexts (tag=1), but you can specify other tags if you wish
def list_of_target_contexts(contexts, interesting_tokens, tag=1):
    token_contexts = []
    for doc in contexts:
        for idx, tok in enumerate(doc):
            if (interesting_tokens is None or tok in interesting_tokens):
                token_contexts.append(create_new_instance(tok, idx, target=doc, label=tag))
    return token_contexts

# create a list of context objects with custom number of elements
# representations - generator
def list_of_contexts_additional(contexts, representations, repr_labels, interesting_tokens, tag=1):
    token_contexts = []
#    representations = [ open(r) for r in representations if type(r) == 'file' ]
    #print representations
    for all_features in zip(contexts, *representations):
        if len(all_features)-1 != len(repr_labels):
            sys.stderr.write( "Wrong number of additional element labels\n" )
            print all_features, repr_labels
            return []
        # target sentence
        doc = all_features[0]

        for idx, tok in enumerate(doc):
            if (interesting_tokens is None or tok in interesting_tokens):
                token_contexts.append(create_new_instance_additional(tok, idx, target=doc, label=tag, elements=all_features[1:], elem_labels=repr_labels))

    for r in representations:
        if type(r) == 'file':
            r.close()   

    return token_contexts


# generate a unique random string
def get_random_name(prefix=''):
    return 'tmp'+prefix+str(time.time())


# tag the input file with Tree Tagger
# returns label for context object and corpus iterator
def get_pos_tagging( src_file, tagger, par_file, label ):
    print "Start tagging", src_file

    # tokenize and add the sentence end marker 
    # tokenization is done with nltk
    tmp_tokenized_name = get_random_name(prefix='_tok')
    tmp_tok = open(tmp_tokenized_name, 'wr+')
    for line in open(src_file):
        words = word_tokenize( line[:-1].decode('utf-8') )
        tmp_tok.write('%s\nSentenceEndMarker\n' % '\n'.join([w.encode('utf-8') for w in words]))
    tmp_tok.seek(0)

    # pass to tree-tagger
    tmp_tagged_name = get_random_name(prefix='_tag')
    tmp_tagged = open(tmp_tagged_name, 'wr+')
    tagger_call = Popen([tagger, '-token', par_file], stdin=tmp_tok, stdout=tmp_tagged)
    tagger_call.wait()
    tmp_tagged.seek(0)

    # remove sentence markers, restore sentence structure
    tmp_final_name = get_random_name(prefix='_final')
    tmp_final = open(tmp_final_name, 'w')
    output = []
    cur_sentence = []
    for line in tmp_tagged:
        word_tag = line[:-1].decode('utf-8').strip().split('\t')
        # each string has to be <word>\t<tag>
        if len(word_tag) != 2:
            continue
        if word_tag[0] == 'SentenceEndMarker':
            tmp_final.write('%s\n' % ' '.join([tag.encode('utf-8') for tag in cur_sentence]))
            output.append(cur_sentence)
            cur_sentence = []
        else:
            cur_sentence.append( word_tag[1] )
    tmp_tok.close()
    tmp_tagged.close()
    tmp_final.close()

    # delete all temporary files
    call(['rm', tmp_tokenized_name, tmp_tagged_name])

    return (label, output)


# force alignment with fastalign
# if no alignment model provided, builds the alignment model first
# <align_model> - path to alignment model such that <align_model>.frd_params, .rev_params, .fwd_err, .rev_err exist
# <src_file>, <tg_file> - files to be aligned
# returns: list of lists of possible alignments for every target word:
#    [ [ [0], [1,2], [], [3,4], [3,4], [7], [6], [] ... ]
#      [ ....                                           ]
#        ....
#      [ ....                                           ] ]
def get_alignments(src_file, tg_file, align_model = None, src_train='', tg_train='', label='alignments'):
    alignments = []
    print "Get alignments"
    if align_model == None:
        print "Train an alignment model"
        cdec = os.environ['CDEC_HOME']
        if cdec == '':
            sys.stderr.write('No CDEC_HOME variable found. Please install cdec and/or set the variable\n')
            return []
        if src_train == '' or tg_train == '':
            sys.stderr.write('No parallel corpus for training\n')
            return []
        # join source and target files
        joint_name = os.path.basename(src_train)+'_'+os.path.basename(tg_train)
        src_tg_file=open(joint_name, 'w')
        get_corp = Popen([cdec+'/corpus/paste-files.pl', src_train, tg_train], stdout=src_tg_file)
        get_corp.wait()
        src_tg_file.close()

        src_tg_clean = open(joint_name+'.clean', 'w')
        clean_corp = Popen([cdec+'/corpus/filter-length.pl', joint_name], stdout=src_tg_clean)
        clean_corp.wait()
        src_tg_clean.close()

        # train the alignment model
        align_model = 'align_model'
        fwd_align = open('align_model.fwd_align', 'w')
        rev_align = open('align_model.rev_align', 'w')
        fwd_err = open('align_model.fwd_err', 'w')
        rev_err = open('align_model.rev_err', 'w')
     
        fwd = Popen([cdec+'/word-aligner/fast_align', '-i'+joint_name+'.clean', '-d', '-v', '-o', '-palign_model.fwd_params'], stdout=fwd_align, stderr=fwd_err )
        rev = Popen([cdec+'/word-aligner/fast_align', '-i'+joint_name+'.clean', '-r', '-d', '-v', '-o', '-palign_model.rev_params'], stdout=rev_align, stderr=rev_err)
        fwd.wait()
        rev.wait()
        
        fwd_align.close()
        rev_align.close()
        fwd_err.close()
        rev_err.close()

    aligner = Aligner(align_model+'.fwd_params',align_model+'.fwd_err',align_model+'.rev_params',align_model+'.rev_err')
    src = open(src_file)
    tg = open(tg_file)
    for src_line, tg_line in zip(src, tg):
        align_str = aligner.align( src_line[:-1].decode('utf-8')+u' ||| '+tg_line[:-1].decode('utf-8') )
        cur_alignments = [ [] for i in range(len(tg_line.split())) ]
        for pair in align_str.split():
            pair = pair.split('-')
            cur_alignments[int(pair[1])].append( pair[0] )
        alignments.append(cur_alignments)
    src.close()
    tg.close()
    
    aligner.close()
    return (label, alignments)


def list_of_bad_contexts(contexts, labels, interesting_tokens=None):
    token_contexts = []
    for doc in contexts:
        label_list = [unicode(l) for l in labels.next()]
        for (idx, (tok,label)) in enumerate(zip(doc,label_list)):
            if (interesting_tokens is None or tok in interesting_tokens) and label == 'B':
                token_contexts.append(create_new_instance(tok, idx, target=doc, label=0))
    return token_contexts


def parse_corpus_contexts(corpus_file, interesting_tokens=None, tag=1):
    corpus = SimpleCorpus(corpus_file)
    return list_of_target_contexts(corpus.get_texts(), interesting_tokens, tag=tag)


# parse_corpus_contexts with additional representations
# <additional> list of representations, each of format (<label>, <generator>)
#def parse_corpus_contexts_additional(corpus_file, tag=1, interesting_tokens=None, *additional):
def parse_corpus_contexts_additional(corpus_file, interesting_tokens, tag, *additional):
    corpus = SimpleCorpus(corpus_file)
    representations = [ r[1] for r in additional ]
    repr_labels = [ r[0] for r in additional ]

    contexts = list_of_contexts_additional(corpus.get_texts(), representations, repr_labels, interesting_tokens, tag=1)

    return contexts


#extract negative contexts from back-translated corpus
def parse_back_translation(corpus_file, labels_file, interesting_tokens=None):
    corpus = SimpleCorpus(corpus_file)
    labels = SimpleCorpus(labels_file)
    return list_of_bad_contexts(corpus.get_texts(), labels.get_texts(), interesting_tokens)

from itertools import groupby
import codecs
# TODO: add support for including the source sentence into the output (takes another file as input)
# matching sentences may require us to include the sen id
def parse_wmt14_data(corpus_file, interesting_tokens=None):
     # recover sentences from a .tsv with senids and tokens (wmt14 format)
    def group_by_senid(filename):
        rows = []
        for l in codecs.open(filename, encoding='utf8'):
            rows.append(l.rstrip().split('\t'))

        sens = []
        # group by sentence id and order by word index so that we can extract the contexts
        for key, group in groupby(rows, lambda x: x[0]):
            sen = list(group)
            sens.append(sen)
        return sens

    def extract_word_exchange_format(wmt_contexts, interesting_tokens=None):
        word_exchange_format = []
        for context in wmt_contexts:
            target_sen = [w[2] for w in context]
            for row in context:
                idx = int(row[1])
                word = row[2]
                tag = row[5]
                if interesting_tokens is not None:
                    if word in interesting_tokens:
                        word_exchange_format.append({'token': word, 'target': target_sen, 'index': idx, 'tag': tag})
                else:
                    word_exchange_format.append({'token': word, 'target': target_sen, 'index': idx, 'tag': tag})
        return word_exchange_format

    sen_groups = group_by_senid(corpus_file)
    wef = extract_word_exchange_format(sen_groups)
    return wef

# semeval format

# A parser takes some input, and returns a list of contexts in the format:  { 'token': <token>, index: <idx>, 'source': [<source toks>]', 'target': [<target toks>], 'tag': <tag>}
# semeval input looks like: <sen1>TAB<sen2>
# the scores are in a separate *.gs.* file
import re
import nltk
from nltk.corpus import stopwords
english_stops = stopwords.words('english')
def parse_semeval(inputfile, scoresfile, stops=False):
    # this code taken from the takelab 2012 STS framework
    def fix_compounds(a, b):
        sb = set(x.lower() for x in b)

        a_fix = []
        la = len(a)
        i = 0
        while i < la:
            if i + 1 < la:
                comb = a[i] + a[i + 1]
                if comb.lower() in sb:
                    a_fix.append(a[i] + a[i + 1])
                    i += 2
                    continue
            a_fix.append(a[i])
            i += 1
        return a_fix

    def load_data(path, scores):
        scores = [float(x) for x in open(scores)]
        lines = list(open(path))
        training_data = []
        assert len(scores) == len(lines), 'the scores file and the text file should have the same number of lines'
        r1 = re.compile(r'\<([^ ]+)\>')
        r2 = re.compile(r'\$US(\d)')
        for (l, score) in zip(open(path), scores):
            l = l.decode('utf-8')
            l = l.replace(u'’', "'")
            l = l.replace(u'``', '"')
            l = l.replace(u"''", '"')
            l = l.replace(u"—", '--')
            l = l.replace(u"–", '--')
            l = l.replace(u"´", "'")
            l = l.replace(u"-", " ")
            l = l.replace(u"/", " ")
            l = r1.sub(r'\1', l)
            l = r2.sub(r'$\1', l)
            if stops:
                sa, sb = tuple(nltk.word_tokenize(s) for s in l.strip().split('\t'))
                sa = [w for w in sa if w not in english_stops]
                sb = [w for w in sb if w not in english_stops]
            else:
                sa, sb = tuple(nltk.word_tokenize(s) for s in l.strip().split('\t'))
            sa, sb = ([x.encode('utf-8') for x in sa],
                      [x.encode('utf-8') for x in sb])

            for s in (sa, sb):
                for i in xrange(len(s)):
                    if s[i] == "n't":
                        s[i] = "not"
                    elif s[i] == "'m":
                        s[i] = "am"
            sa, sb = fix_compounds(sa, sb), fix_compounds(sb, sa)
            training_data.append({'source': sa, 'target': sb, 'tag': score})
        return training_data

    return load_data(inputfile, scoresfile)

