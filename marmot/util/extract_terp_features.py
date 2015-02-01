import argparse
import os
import sys
from subprocess import Popen, PIPE, call

# import prepare_dataset

cdec_home = ""
prefix = ""


#Add sentence ids for TERp
def add_num(file_name):
    f_in = open(file_name)
    f_out = open(get_name(file_name,'num'),'w')

    cnt = 0
    max_digits = 6
    for line in f_in:
        line = line[:-1].decode('utf-8')
        cur_num = str(cnt)
        while len(cur_num) < max_digits:
            cur_num = '0'+cur_num
        f_out.write("%s (%s)\n" % (line.encode('utf-8'), cur_num))
        cnt += 1

    f_in.close()
    f_out.close()


#generate new filename with a specified extension
def get_name(file_name, extension):
    return file_name[:file_name.rfind('.')+1]+extension


#parse a set of lines from .pra file that belong to one sentence
def parse_ter_instance(lines):
    sent_id, align, loc_map = "","",""
    for line in lines:
        stop = line.find(':')
        line_id = line[:stop]
        if line_id == "Sentence ID":
            sent_id = line[stop+2:-1]
        elif line_id == "Alignment":
            align = line[stop+2:-1]
        elif line_id == "HypLocMap":
             loc_map = line[stop+2:-1]

    if not align or not loc_map:
        raise RuntimeError("Unexpected instance format for sentence {0}".format(sent_id))
    if not sent_id:
        raise RuntimeError("Unexpected instance format: no sentence id")
    return sent_id+'\t'+align+'\t'+loc_map


#convert TERp output (.pra) to short format: 
#sentence_id, alignment, HypLocMap -- tab-separated
#returns set of numbers of lines with incorrect TERp output
#the same set is written to the file $DATA_DIR/missing_lines
# TODO: this function writes a file as a side effect -- avoid this
def compress_pra(pra_file_name):
    pra = open(pra_file_name)
    print('pra is: ' + pra.name)
    new_file = open(get_name(pra_file_name, 'pra_short'), 'w')
    print('new_file is: ' + new_file.name)
    missing_lines = open(os.path.dirname(os.path.realpath(pra_file_name))+"/missing_lines", 'w')

    cnt = 0
    all_missing_lines_set = set()
    cur_sentence = []
    for line in pra:
        cur_sentence.append(line.decode("utf-8"))
        if line.startswith('Score: '):
            try:
                print('line starting with Score:')
                print(line)
                new_file.write("%s\n" % parse_ter_instance(cur_sentence))
            # TODO: what error are we catching?
            except:
                missing_lines.write("%d\n" % cnt)
                all_missing_lines_set.add(cnt)

            cur_sentence = []
            cnt += 1

    pra.close()
    new_file.close()

    return all_missing_lines_set

#calls tokenize-anything.sh script from cdec, writes tokenized text to file_name.tok
def tokenize(file_name):
    # TODO: no globals -- get from environment, or pass them into this function
    global cdec_home
    global prefix
    tok_name = file_name+'.tok'

    in_file = open(file_name)
    out_file = open(tok_name,'w')
    p = Popen([cdec_home+"/corpus/tokenize-anything.sh"], stdin=in_file, stdout=out_file)
    p.communicate()
    in_file.close()
    out_file.close()

    add_num(tok_name)

#remove lines from file if they had incorrect TERp output
def delete_lines(file_name, missing_lines):
    cnt = 0
    in_file = open(file_name)
    out_file = open(get_name(file_name, 'short'), 'w')
    for line in in_file:
        line = line[:-1].decode("utf-8")
        if not cnt in missing_lines:
            out_file.write("%s\n" % line.encode("utf-8"))

    in_file.close()
    out_file.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Extract features from post-edited automatic translations')
    parser.add_argument("-a", "--automatic", required=True, help="automatic translations in plain-text format")
    parser.add_argument("-r", "--reference", required=True, help="post-editions in plain-text format")
    parser.add_argument("-t", "--terp", default=os.environ.get('TERP'), help="path to TERp root directory")
    parser.add_argument("-c","--cdec", default=os.environ.get('CDEC_HOME'), help='path to cdec root directory')

    args = parser.parse_args()

    if not args.terp:
        sys.stderr.write("No TERp found. Please provide the path to TERp root directory or specify it in TERP environment variable\n")

    if not args.cdec:
        sys.stderr.write("No cdec found. Please provide the path to cdec root directory or specify it in CDEC_HOME environment variable\n")

    if not args.cdec or not args.terp:
        sys.exit(1)

    cdec_home = args.cdec
    prefix = args.automatic[:args.automatic.rfind('.')]

    #tokenize and add sentence numbers
    sys.stderr.write("Tokenization\n")
    tokenize(args.automatic)
    tokenize(args.reference)

    #TERp annotation
    #TERp writes all results to files, so no answer from subprocess is needed
    sys.stderr.write("Computation of edit distance with TERp\n")
    # this computes TER, not TERp
    # call([args.terp+"/bin/terp_ter", " -r "+args.reference+'.num', " -h "+args.automatic+'.num', " -n "+prefix])
    # try calling the real terp jar
    call(["java", " -jar "+args.terp+"dist/lib/terp.jar", " -r "+args.reference+'.num', " -h "+args.automatic+'.num', " -n "+prefix])
    missing_lines_set = compress_pra(prefix+'.pra')

    #remove the lines which have errors in TERp output and didn't get to .pra_short
    if missing_lines_set:
        delete_lines(args.automatic+".tok", missing_lines_set)
        delete_lines(args.reference+".tok", missing_lines_set)

    sys.stderr.write("TERp output is in %s.pra\n" % prefix)
    sys.stderr.write("Compressed TERp output is in %s\n" % (prefix+".pra_short"))
    if missing_lines_set:
        sys.stderr.write("New hypothesis and reference files are in %s and %s\n" % (args.automatic+".short", args.reference+".short"))
