from __future__ import print_function
import os


# generates a template for crf++ feature extractor: all columns will be used as features, 
# no combinations of columns, no contexts (it should already be in original feature set)
def generate_crf_template(feature_num, template_name='template', tmp_dir='tmp_dir'):
    if not os.path.isdir(tmp_dir):
        print("Wrong temporary directory: ", tmp_dir)
        return
    template = open(os.path.join(tmp_dir, template_name), 'w')
    print("Saving template to ", os.path.join(tmp_dir, template_name))
    template.write('# Unigram\n')
    for i in range(feature_num):
        a_str = 'U{}:%x[0,{}]'.format(i, i)
        template.write('%s\n' % a_str)
    template.write('\n# Bigram\nB')
    template.close()
