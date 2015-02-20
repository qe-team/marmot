# persist features to file
# feature output formats are different depending upon the datatype
# if it's an ndarray, write to .csv
# if it's an list of lists, write to crf++ format, with a separate file containing the feature names
# if it's a dict, write to .json or pickle the object(?), write the feature names to a separate file
import os, errno
import pandas as pd
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('experiment_logger')

# TODO: make sure to check the type of the features
def persist_features(dataset_name, features, persist_dir, feature_names=None):
    '''
    persist the features to persist_dir -- use dataset_name as the prefix for the persisted files
    :param dataset_name:
    :param features:
    :param persist_dir:
    :param feature_names:
    :return:
    '''
    try:
        os.makedirs(persist_dir)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(persist_dir):
            pass
        else:
            raise

    # for the 'plain' datatype
    if type(features) == np.ndarray and features.shape[1] == len(feature_names):
        output_df = pd.DataFrame(data=features, columns=feature_names)
        output_path = os.path.join(persist_dir, dataset_name + '.csv')
        output_df.to_csv(output_path, index=False)
        logger.info('saved features in: {} to file: {}'.format(dataset_name, output_path))

