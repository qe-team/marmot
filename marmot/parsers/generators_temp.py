def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

# copy of the function from parsers.py, but it writes the tagging to the file
# TODO: this is specifically for tree tagger, it's not a general pos tagging function
# TODO: this function should be a preprocessing option -- it's actually a representation generator
# and returns the filename
def get_pos_tagging(src_file, tagger, par_file, label):
    print("Start tagging", src_file)
    # tokenize and add the sentence end marker
    # tokenization is done with nltk
    tmp_tokenized_name = get_random_name(prefix='_tok')
    tmp_tok = open(tmp_tokenized_name, 'wr+')
    for line in open(src_file):
        words = word_tokenize(line[:-1].decode('utf-8'))
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
            cur_sentence.append(word_tag[1])
    tmp_tok.close()
    tmp_tagged.close()
    tmp_final.close()

    # delete all temporary files
    call(['rm', tmp_tokenized_name, tmp_tagged_name])

    return (label, tmp_final_name)


def cur_dir():
    import os
    print(os.getcwd())


# generate a unique random string
# used for temporary file creation and deletion
def get_random_name(prefix=''):
    return 'tmp'+prefix+str(time.time())


# TODO: this is an additional representation - requires the alignment model files to exist
# force alignment with fastalign
# if no alignment model provided, builds the alignment model first
# <align_model> - path to alignment model such that <align_model>.frd_params, .rev_params, .fwd_err, .rev_err exist
# <src_file>, <tg_file> - files to be aligned
# returns: list of lists of possible alignments for every target word:
#    [ [ [0], [1,2], [], [3,4], [3,4], [7], [6], [] ... ]
#      [ ....                                           ]
#        ....
#      [ ....                                           ] ]
def force_alignments(src_file, tg_file, trained_model):
    alignments = []
    aligner = Aligner(trained_model+'.fwd_params',trained_model+'.fwd_err',trained_model+'.rev_params',trained_model+'.rev_err')
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

    return alignments


# copy of the function from parsers.py, but it writes the tagging to the file
# and returns the filename
def get_alignments(src_file, tg_file, trained_model=None, src_train='', tg_train='', align_model='align_model', label='alignments'):
    if trained_model is None:
        trained_model = train_alignments(src_train, tg_train, align_model)
        if trained_model == '':
            sys.stderr.write('No alignment model trained\n')
            return []

    aligner = Aligner(trained_model+'.fwd_params', trained_model+'.fwd_err', trained_model+'.rev_params', trained_model+'.rev_err')
    src = open(src_file)
    tg = open(tg_file)
    align_file = src_file+'_'+os.path.basename(tg_file)+'.aligned'
    aligned = open(align_file, 'w')
    for src_line, tg_line in zip(src, tg):
        aligned.write(aligner.align(src_line[:-1].decode('utf-8')+u' ||| '+tg_line[:-1].decode('utf-8'))+u'\n')
    aligned.close()
    aligner.close()

    return (label, align_file)

