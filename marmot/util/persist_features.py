# persist features to file
# feature output formats are different depending upon the datatype
# if it's a flat list, write to .csv
# if it's a list of lists, write to crf++ format, with a separate file containing the feature names
# if it's a dict, write to .json or pickle the object(?), write the feature names to a separate file

import pandas as pd

# TODO: make sure to check the type of the features
def persist_features(dataset_name, features, persist_dir):
    # persist the features to persist_dir -- use dataset_name as the prefix for the persisted files
    pass
