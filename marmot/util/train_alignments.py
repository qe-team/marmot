import os
from subprocess import Popen

def train_alignments(src_train, tg_train, align_model='align_model'):
    cdec = os.environ['CDEC_HOME']
    if cdec == '':
        sys.stderr.write('No CDEC_HOME variable found. Please install cdec and/or set the variable\n')
        return ''
    if src_train == '' or tg_train == '':
        sys.stderr.write('No parallel corpus for training\n')
        return ''
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
    fwd_align = open(align_model+'.fwd_align', 'w')
    rev_align = open(align_model+'.rev_align', 'w')
    fwd_err = open(align_model+'.fwd_err', 'w')
    rev_err = open(align_model+'.rev_err', 'w')

    fwd = Popen([cdec+'/word-aligner/fast_align', '-i'+joint_name+'.clean', '-d', '-v', '-o', '-palign_model.fwd_params'], stdout=fwd_align, stderr=fwd_err )
    rev = Popen([cdec+'/word-aligner/fast_align', '-i'+joint_name+'.clean', '-r', '-d', '-v', '-o', '-palign_model.rev_params'], stdout=rev_align, stderr=rev_err)
    fwd.wait()
    rev.wait()

    fwd_align.close()
    rev_align.close()
    fwd_err.close()
    rev_err.close()

    return align_model

