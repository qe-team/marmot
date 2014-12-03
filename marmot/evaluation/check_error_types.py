
from __future__ import division, print_function
import codecs
import argparse
from itertools import groupby

def get_error_distribution(error_file):
    with codecs.open(error_file, encoding='utf8') as tsv:
        # remove newlines
        row_data = [ l.rstrip().split('\t') for l in tsv ]
        class_data = [ {'predicted': l[0], 'actual': l[1], 'word': l[2], 'class': l[3] } for l in row_data ]

    class_data = sorted(class_data, key=lambda x: x['class'])
    for key, group in groupby(class_data, lambda x: x['class']):
        group_instances = list(group)
        print('ERROR TYPE: {}\tTOTAL INSTANCES: {}'.format(key, str(len(group_instances))))
        accuracy = sum([1 for i in group_instances if i['predicted'] == i['actual']]) / len(group_instances)
        print("group accuracy: {}".format(str(accuracy)))
        # for i in group_instances:
        #     print(i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='input file -- sentences tagged with errors')
    args = parser.parse_args()
    get_error_distribution(args.input)