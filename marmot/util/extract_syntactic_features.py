# Extract syntactic sentence-level features from the output of Stanford parser
# Parser should be run using the following command:
#
# for English:
# java -mx3g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLP -file <INPUT> -outputFormat xml -annotators tokenize,ssplit,pos,depparse
# for German:
# java -mx3g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLP -props StanfordCoreNLP-german.properties -file <INPUT> -outputFormat xml -annotators tokenize,ssplit,pos,depparse
#
# in directory /export/tools/varvara/stanford_compiled/stanford-corenlp-full-2015-01-30
from __future__ import print_function, division
import xml.etree.ElementTree as ET
from subprocess import call
import numpy as np
import time
import sys
import os


def call_stanford(data_src, tmp_dir):
    cur_dir = os.getcwd()
    # call Stanford
    os.chdir('/export/tools/varvara/stanford_compiled/stanford-corenlp-full-2015-01-30')
    sys.stderr.write('Changed to Stanford dir\n')
    sys.stderr.write('Cur dir: %s\n' % os.getcwd())
    parsed_src_name = os.path.join(tmp_dir, os.path.basename(data_src) + '.xml')
    sys.stderr.write('Output file will be: %s\n' % parsed_src_name)
    syntactic_command_src = "java -Xmx10g -mx3g -cp '*' edu.stanford.nlp.pipeline.StanfordCoreNLP -file %s -outputFormat xml -annotators tokenize,ssplit,pos,depparse -ssplit.eolonly true -outputDirectory %s" % (data_src, tmp_dir)
    # write syntactic command to file
    command_file_name = 'tagger_run.' + str(time.time())
    sys.stderr.write('Parsing command:\n%s\n' % syntactic_command_src)
    command_file = open(command_file_name, 'w')
    command_file.write('%s\n' % syntactic_command_src)
    command_file.close()
    call(['bash', command_file_name])
    os.remove(command_file_name)

    os.chdir(cur_dir)
    return parsed_src_name


def call_parzu(data_tg, tmp_dir):
    cur_dir = os.getcwd()
    # call ParZu
    os.chdir('/export/tools/varvara/ParZu')
    syntactic_command_tg = "./parzu -i tokenized_lines"
    parsed_tg_name = os.path.join(tmp_dir, os.path.basename(data_tg) + '.parzu')
    parsed_tg = open(parsed_tg_name, 'w')
    call(syntactic_command_tg.split(), stdin=open(data_tg), stdout=parsed_tg)
    parsed_tg.close()
    os.chdir(cur_dir)
    return parsed_tg_name


def go_down(idx, cur_dep, dependencies):
    if idx not in dependencies:
        return cur_dep
    return get_depth(dependencies, idx, cur_dep)


def get_depth(dependencies, root, cur_dep):
    max_dep = 0
    for arrow in dependencies[root]:
        new_dep = go_down(arrow['id'], cur_dep + 1, dependencies)
        max_dep = new_dep if new_dep > max_dep else max_dep
    return max_dep


# return: maximum depth of the tree,
#         average depth
#         proportion of internal nodes
def get_paths(dependencies, sentence_id):
    all_paths = []

    # find list of all tokens and root
    tokens = set()          # list of all tokens
    dep_tokens = []         # tokens that have dependencies
    internal_tokens = []    # internal nodes of the tree
    for t in dependencies:
        tokens.add(t)
        internal_tokens.append(t)
        for dep in dependencies[t]:
            tokens.add(dep['id'])
            dep_tokens.append(dep['id'])
    tokens = list(tokens)
    root_list = [t for t in tokens if t not in dep_tokens]
    assert(len(root_list) == 1), "Wrong number of roots: {}, sentence_id: {}, dependencies: {}".format(len(root_list), sentence_id, dependencies)
    root = root_list[0]
    internal_tokens.remove(root)

    # all paths
    for t in tokens:
        # use only leaves - tokens with no outcoming dependencies
        if t in dependencies:
            continue
        cur_dep = 0
        cur_head = t
        while cur_head != root:
            for head in dependencies:
                for dep in dependencies[head]:
                    if dep['id'] == cur_head:
                        cur_dep += 1
                        cur_head = head
        all_paths.append(cur_dep)
    try:
        features = [max(all_paths), np.average(all_paths), len(internal_tokens)/len(tokens)]
    except ValueError as ex:
        sys.stderr.write('%s\n' % ' '.join([str(e) for e in ex]))
        sys.stderr.write("In sentence %s\n" % (sentence_id))
        sys.exit()
    return features


def get_width(dependencies, root, sentence_id):
    assert(root in dependencies), "Wrong root: {}, dependencies: {}, sentence: {}".format(root, dependencies, sentence_id)
    return len(dependencies[root])


def get_connection_features(dependencies, token_pos, language=None):
    # clauses inventory
    clauses_en = ['advcl', 'ccomp', 'pcomp', 'rcmod']
    clauses_de = ['neb', 'objc', 'par', 'rel']
    if language == 'en':
        clauses = clauses_en
    elif language == 'de':
        clauses = clauses_de
    else:
        clauses = clauses_en + clauses_de
    # subjects inventory
    subject = ['nsubj', 'nsubjpass', 'subj']
    verbs = 'V'
    # number of subjects, number of verbs with dependent subject, number of dependent clauses
    n_subj, n_verb_subj, n_clauses = 0, 0, 0
    for head in dependencies:
        for dep in dependencies[head]:
            if dep['type'] in clauses:
                n_clauses += 1
            if dep['type'] in subject:
                n_subj += 1
                if token_pos[head].startswith(verbs):
                    n_verb_subj += 1
    return [n_subj, n_verb_subj, n_clauses]


def get_pos(token_pos):
    # POS tags for verbs, nouns, conjunctions
    verbs = 'V'
    nouns = 'N'
    conjunctions = 'CC'
    conj_de = 'konj'
    # number of verbs, nouns, conjunctions
    n_verbs, n_nouns, n_conj = 0, 0, 0
    # sentence starts with a verb
    start_verb = 0
    for t in token_pos:
        token_str = token_pos[t]
        if token_str.startswith(verbs):
            n_verbs += 1
        if token_str.startswith(nouns):
            n_nouns += 1
        if token_str.startswith(conjunctions):
            n_conj += 1
        if token_str.startswith(conj_de):
            n_conj += 1
    if token_pos[1].startswith(verbs):
        start_verb = 1
    return [n_verbs, n_nouns, n_conj, start_verb]


# this procedure parses Stanford CoNLL output
# ROOT is a nonterminal
# which has one dependant -- actual root (predicate)
# all other words are connected to the terminal root
def parse_xml(xml_output):
    tree = ET.parse(xml_output)
    root_el = tree.getroot()
    sentences = []
    for idx, sent in enumerate(root_el.getchildren()[0].getchildren()[0].getchildren()):
        if idx % 1000 == 0:
            sys.stderr.write('.')
        sent_tokens = {}
        sent_token_pos = {}
        sent_dependencies = {}
        sent_root = None
        sent_id = sent.attrib['id']
        for field in sent.getchildren():
            # get tokens and their POS
            if field.tag == 'tokens':
                for token in field.getchildren():
                    word_id = int(token.attrib['id'])
                    for tok_field in token.getchildren():
                        if tok_field.tag == 'word':
                            sent_tokens[word_id] = tok_field.text
                        elif tok_field.tag == 'POS':
                            sent_token_pos[word_id] = tok_field.text
            # parse dependencies
            elif field.tag == 'dependencies' and field.attrib['type'] == 'basic-dependencies':
                for dep in field.getchildren():
                    d_type = dep.attrib['type']
                    d_head, d_child = None, None
                    for d_field in dep.getchildren():
                        if d_field.tag == 'governor':
                            d_head = int(d_field.attrib['idx'])
                        elif d_field.tag == 'dependent':
                            d_child = int(d_field.attrib['idx'])
                    if d_head is None or d_child is None:
                        sys.stderr.write("Wrong dependency format\n")
                        sys.exit()
                    if d_head == 0:
                        sent_root = d_child
                        sent_dependencies[d_child] = []
                    else:
                        if d_head-1 not in sent_dependencies:
                            sent_dependencies[d_head-1] = []
                        sent_dependencies[d_head-1].append({'id': d_child-1, 'type': d_type})
        sentences.append({'tokens': sent_tokens, 'pos': sent_token_pos, 'dependencies': sent_dependencies, 'root': sent_root, 'id': sent_id})
    return sentences


# this is valid only for parsing the output of ParZu
# it generates a tree where ROOT is a nonterminal
# all predicates and punctuation marks are connected to it
def parse_conll(conll_file):
    sentences = []
    sent_tokens = {}
    sent_token_pos = {}
    sent_dependencies = {}
    sent_root = 0
    sent_id = 0
    for line in open(conll_file):
        if line == '\n' and len(sent_tokens) > 0:
            sentences.append({'tokens': sent_tokens, 'pos': sent_token_pos, 'dependencies': sent_dependencies, 'root': sent_root, 'id': sent_id})
            sent_tokens = {}
            sent_token_pos = {}
            sent_dependencies = {}
            sent_id += 1
            continue
        chunks = line.decode('utf-8').strip('\n').split('\t')
        word_id = int(chunks[0])
        sent_tokens[word_id] = chunks[1]
        sent_token_pos[word_id] = chunks[4]
        sent_root = 0
        d_head = int(chunks[6])
        if d_head-1 not in sent_dependencies:
            sent_dependencies[d_head-1] = []
        sent_dependencies[d_head-1].append({'id': word_id-1, 'type': chunks[7]})
    if len(sent_tokens) > 0:
        sentences.append({'tokens': sent_tokens, 'pos': sent_token_pos, 'dependencies': sent_dependencies, 'root': sent_root, 'id': sent_id})
    return sentences


def features_one_lang(sentences, language=None):
    all_features = []
    for idx, sent in enumerate(sentences):
        if idx % 100 == 0:
            sys.stderr.write('.')
        sent_features = []
        sent_features.extend(get_paths(sent['dependencies'], sent['id']))
        sent_features.append(get_width(sent['dependencies'], sent['root'], sent['id']))
        sent_features.extend(get_connection_features(sent['dependencies'], sent['pos'], language=language))
        sent_features.extend(get_pos(sent['pos']))
        all_features.append(sent_features)
    return all_features


def extract_syntactic_features(file_src, file_tg, output_file, ext_src='xml', ext_tg='conll'):
    sys.stderr.write('Parse source file %s\n' % file_src)
    if ext_src == 'xml':
        sentences_src = parse_xml(file_src)
    elif ext_src == 'conll':
        sentences_src = parse_conll(file_src)
    sys.stderr.write('Parse target file %s\n' % file_tg)
    if ext_tg == 'xml':
        sentences_tg = parse_xml(file_tg)
    elif ext_tg == 'conll':
        sentences_tg = parse_conll(file_tg)

    sys.stderr.write("Extract source features\n")
    all_features_src = features_one_lang(sentences_src, language='en')
    sys.stderr.write("Extract target features\n")
    all_features_tg = features_one_lang(sentences_tg, language='de')

    sys.stderr.write("Write syntactic features\n")
    output = open(output_file, 'w')
    for feat_src, feat_tg in zip(all_features_src, all_features_tg):
        output.write('%s\t%s\n' % ('\t'.join([str(f) for f in feat_src]), '\t'.join([str(f) for f in feat_tg])))


if __name__ == "__main__":
    language = None
    if len(sys.argv) == 3:
        language = sys.argv[2]
    sys.stderr.write("Language -- %s\n" % str(language))
    if sys.argv[1].endswith('xml'):
        sys.stderr.write("Parsing xml file %s" % sys.argv[1])
        sentences = parse_xml(sys.argv[1])
    else:
        sys.stderr.write("Parsing conll file %s" % sys.argv[1])
        sentences = parse_conll(sys.argv[1])
    sys.stderr.write("Parsing finished")
    all_features = []
    for idx, sent in enumerate(sentences):
        if idx % 100 == 0:
            sys.stderr.write('.')
        sent_features = []
        sent_features.extend(get_paths(sent['dependencies'], sent['id']))
        sent_features.append(get_width(sent['dependencies'], sent['root'], sent['id']))
        sent_features.extend(get_connection_features(sent['dependencies'], sent['pos'], language=language))
        sent_features.extend(get_pos(sent['pos']))
        all_features.append(sent_features)
    for feat_list in all_features:
        sys.stdout.write('%s\n' % '\t'.join([str(f) for f in feat_list]))
