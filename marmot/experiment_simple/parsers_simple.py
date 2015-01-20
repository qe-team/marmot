import os
import time
from nltk import word_tokenize
from subprocess import call, Popen

from marmot.util.force_align import Aligner
from marmot.util.alignments import train_alignments
# this is not to make parsers.py too big
# should probably be re-organized


# convert WMT file to two files: plain text and labels
#    plain text:  word word word word word
#    labels:      OK   BAD  BAD  OK   OK
# check that source and target have the same number of sentences, re-write source file without sentence ids
def parse_wmt_to_text( wmt_file, wmt_source_file ):
    print "Parse wmt" 
    tmp_dir = os.getcwd()+'/tmp_dir'
    call(['mkdir', '-p', tmp_dir])

    target_file = tmp_dir+'/'+os.path.basename(wmt_file)+'.target'
    tags_file = tmp_dir+'/'+os.path.basename(wmt_file)+'.tags'
    source_file = tmp_dir+'/'+os.path.basename(wmt_source_file)+'.txt'

    target = open(target_file, 'w')
    tags = open(tags_file, 'w')
    source = open(source_file, 'w')
    cur_num = '0'
    cur_sent = []
    cur_tags = []

    source_sents = {}
    for line in open(wmt_source_file):
        str_num = line.decode('utf-8').strip().split('\t')
        source_sents[str_num[0]] = str_num[1]

    for line in open(wmt_file):
        chunks = line[:-1].decode('utf-8').split('\t')
        if chunks[0] != cur_num:
            if cur_sent != []:
                if source_sents.has_key(cur_num):
                    source.write('%s\n' % source_sents[cur_num].encode('utf-8'))
                    target.write('%s\n' % (' '.join([w.encode('utf-8') for w in cur_sent]) ))
                    tags.write('%s\n' % (' '.join([w.encode('utf-8') for w in cur_tags])) )
                cur_sent = []
                cur_tags = []
            cur_num = chunks[0]
        else:
           cur_sent.append(chunks[2])
           cur_tags.append(chunks[5])
    tags.close()
    target.close()
    print "THREE FILES", target_file, source_file, tags_file
    return { 'target': target_file, 'source': source_file, 'tag': tags_file }

def get_corpus( target_file, source_file, tag ):
#    if type(tag) == int:
     return { 'target': target_file, 'source': source_file, 'tag': tag }
#    elif os.path.isfile(tag):
#        return { 'target': target_file, 'source': source_file, 'tag_file': tag }
#    else:
#        print "Tag "+tag+" has invalid type: should be integer or file"
#        return {}


def cur_dir():
    import os
    print os.getcwd()


# generate a unique random string
def get_random_name(prefix=''):
    return 'tmp'+prefix+str(time.time())


# copy of the function from parsers.py, but it writes the tagging to the file 
# and returns the filename
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

    return (label, tmp_final_name)


# copy of the function from parsers.py, but it writes the tagging to the file
# and returns the filename
def get_alignments(src_file, tg_file, trained_model = None, src_train='', tg_train='', align_model = 'align_model', label='alignments'):
    alignments = []
    print "Get alignments"
    if trained_model == None:
        trained_model = train_alignments(src_train, tg_train, align_model)
        if trained_model == '':
            sys.stderr.write('No alignment model trained\n')
            return []

    print 'Trained model: ', trained_model

    aligner = Aligner(trained_model+'.fwd_params',trained_model+'.fwd_err',trained_model+'.rev_params',trained_model+'.rev_err')
    src = open(src_file)
    tg = open(tg_file)
    align_file = src_file+'_'+os.path.basename(tg_file)+'.aligned'
    aligned = open(align_file, 'w')
    for src_line, tg_line in zip(src, tg):
        aligned.write( aligner.align( src_line[:-1].decode('utf-8')+u' ||| '+tg_line[:-1].decode('utf-8') )+u'\n' )
    aligned.close()

    return (label, align_file)


def create_context( repr_dict ):
    context_list = []
    # is checked before in create_contexts, but who knows
    if not repr_dict.has_key('target'):
        print "No 'target' label in data representations"
        return []
    for idx, word in enumerate(repr_dict['target']):
        c = {}
        c['token'] = word
        c['index'] = idx
        for k in repr_dict.keys():
            c[k] = repr_dict[k]
        context_list.append(c)
    return context_list


# create context objects from a data_obj: a dictionary with representation labels as keys ('target', 'source', etc.) and files as values
# output: if sequences = False, one list of context objects is returned
#         if sequences = True, list of lists of context objects is returned (list of sequences)
def create_contexts( data_obj, sequences=False ):
    contexts = []
    if not data_obj.has_key('target'):
        print "No 'target' label in data representations"
        return []
    corpora = [ SimpleCorpus(d) for d in data_obj.values() ]
    for sents in zip(*[c.get_texts() for c in corpora]):
        if sequences:
            contexts.append( create_context( { data_obj.keys()[i]:sents[i] for i in range(len(sents)) } ) )
        contexts.extend( create_context( { data_obj.keys()[i]:sents[i] for i in range(len(sents)) } ) )








 
