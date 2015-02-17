#!/usr/bin/python
# -*- coding: utf-8 -*-

# A parser takes some input, and returns a list of contexts in the format:  { 'token': <token>, index: <idx>, 'source': [<source toks>]', 'target': [<target toks>], 'tag': <tag>}
# return a context object from an iterable of contexts, and a set of interesting tokens
from marmot.util.simple_corpus import SimpleCorpus
from collections import defaultdict, Counter
from subprocess import Popen, call
from nltk import word_tokenize
import time
import os, sys, errno

# TODO: these are for generating the representation
from marmot.util.force_align import Aligner
from marmot.util.alignments import train_alignments


# TODO: this belongs in utils
def extract_important_tokens(corpus_file, min_count=1):
    corpus = SimpleCorpus(corpus_file)
    word_counts = defaultdict(int)
    for context in corpus.get_texts():
        for word in context:
            word_counts[word] += 1
    return set([k for k,v in word_counts.items() if v >= min_count])


# extract important tokens from WMT test format
def extract_important_tokens_wmt(corpus_file, min_count=1):
    all_words = []
    for line in open(corpus_file):
        all_words.append(line.decode('utf-8').split('\t')[2])
    word_counts = Counter(all_words)
    return set([k for k,v in word_counts.items() if v > min_count])

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
            if interesting_tokens is None or tok in interesting_tokens:
                token_contexts.append(create_new_instance(tok, idx, target=doc, label=tag))
    return token_contexts

# create a list of context objects with custom number of elements
# representations - generator
def list_of_contexts_additional(contexts, representations, repr_labels, interesting_tokens, tag=1):
    token_contexts = []
    for all_features in zip(contexts, *representations):
        if len(all_features)-1 != len(repr_labels):
            sys.stderr.write( "Wrong number of additional element labels\n" )
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


# get all of the bad contexts in a list of contexts
def list_of_bad_contexts(contexts, labels, interesting_tokens=None):
    token_contexts = []
    for doc in contexts:
        label_list = [unicode(l) for l in labels.next()]
        for (idx, (tok,label)) in enumerate(zip(doc,label_list)):
            if (interesting_tokens is None or tok in interesting_tokens) and label == 'B':
                token_contexts.append(create_new_instance(tok, idx, target=doc, label=0))
    return token_contexts

# TODO this is for the word-specific classifiers
def parse_corpus_contexts(corpus_file, interesting_tokens=None, tag=1):
    corpus = SimpleCorpus(corpus_file)
    return list_of_target_contexts(corpus.get_texts(), interesting_tokens, tag=tag)


def get_corpus_file(corpus_file, label):
    corpus = SimpleCorpus(corpus_file)
    return (label, corpus.get_texts())

# convert WMT file to two files: plain text and labels
#    plain text:  word word word word word
#    labels:      OK   BAD  BAD  OK   OK
# check that source and target have the same number of sentences, re-write source file without sentence ids
def parse_wmt_to_text(wmt_file, wmt_source_file):
    tmp_dir = os.getcwd()+'/tmp_dir'
    mkdir_p(tmp_dir)

    # TODO: the next three lines are critical, because the alignment representation generator depends upon them being there when it runs
    # TODO: make the persisting of these files explicit and parameterized
    target_file = tmp_dir+'/'+os.path.basename(wmt_file)+'.target'
    tags_file = tmp_dir+'/'+os.path.basename(wmt_file)+'.tags'
    source_file = tmp_dir+'/'+os.path.basename(wmt_source_file)+'.txt'

    target = open(target_file, 'w')
    tags = open(tags_file, 'w')
    source = open(source_file, 'w')
    cur_num = None
    cur_sent, cur_tags = [], []

    # parse source files
    source_sents = {}
    for line in open(wmt_source_file):
        str_num = line.decode('utf-8').strip().split('\t')
        source_sents[str_num[0]] = str_num[1]

    # parse target file and write new source, target, and tag files
    for line in open(wmt_file):
        chunks = line[:-1].decode('utf-8').split('\t')
        if chunks[0] != cur_num:
            if cur_num is not None:
                # check that the sentence is in source
                if cur_num in source_sents:
                    source.write('%s\n' % source_sents[cur_num].encode('utf-8'))
                    target.write('%s\n' % (' '.join([w.encode('utf-8') for w in cur_sent])))
                    tags.write('%s\n' % (' '.join([w.encode('utf-8') for w in cur_tags])))
                cur_sent = []
                cur_tags = []
            cur_num = chunks[0]
        cur_sent.append(chunks[2])
        cur_tags.append(chunks[5])
    # last sentence
    if len(cur_sent) > 0 and cur_num in source_sents:
        source.write('%s\n' % source_sents[cur_num].encode('utf-8'))
        target.write('%s\n' % (' '.join([w.encode('utf-8') for w in cur_sent])))
        tags.write('%s\n' % (' '.join([w.encode('utf-8') for w in cur_tags])))

    tags.close()
    target.close()
    source.close()

    return {'target': target_file, 'source': source_file, 'tag': tags_file}

# TODO: parse here, don't return the file names
def get_corpus(target_file, source_file, tag):
    return {'target': target_file, 'source': source_file, 'tag': tag}


from itertools import groupby
import codecs

# matching sentences may require us to include the sen id
# if source is not provided, pass an empty string ('') as <source_file>
def parse_wmt14_data(corpus_file, source_file, interesting_tokens=None):
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


    def extract_word_exchange_format(wmt_contexts, source=None, interesting_tokens=None):
        word_exchange_format = []
        for i,context in enumerate(wmt_contexts):
            target_sen = [w[2] for w in context]
            source_sen = source[i] if source else None
            for row in context:
                obj_items = []
                word = row[2]
                obj_items.append(('index', int(row[1])))
                obj_items.append(('token', word))
                obj_items.append(('tag', row[5]))
                obj_items.append(('target', target_sen))
                if source:
                    obj_items.append(('source', source_sen))
                if not interesting_tokens or word in interesting_tokens:
                        word_exchange_format.append({ k:val for (k, val) in obj_items })
        return word_exchange_format

    sen_groups = group_by_senid(corpus_file)
    source_sen_groups = None
    if source_file != '':
        source_sen_groups = [ word_tokenize( line[:-1].split('\t')[1] ) for line in codecs.open(source_file, encoding='utf-8') ]
    wef = extract_word_exchange_format(sen_groups, source=source_sen_groups, interesting_tokens=interesting_tokens)
    return wef


# semeval format

# A parser takes some input, and returns a list of contexts in the format:  { 'token': <token>, index: <idx>, 'source': [<source toks>]', 'target': [<target toks>], 'tag': <tag>}
# semeval input looks like: <sen1>TAB<sen2>
# the scores are in a separate *.gs.* file
# TODO: this currently removes stopwords by default (despite the stops=False)
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
        lines = list(open(path))
        if scores is not None:
            scores = [float(x) for x in open(scores)]
        else:
            scores = [0. for i in range(len(lines))]
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

