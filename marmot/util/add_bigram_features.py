def add_bigram_features(features, labels):
    '''
    Enhance feature set with features that consist 
    of a feature + label of previous word

    E.g. from a set of features ['NN', 'Noun', 3]
    create a set ['NN_OK', 'Noun_OK', '3_OK']
    '''

    assert(len(features) == len(labels))
    new_features = []
    for feat_element, a_label in zip(features, labels):
        new_feat_element = []
        for a_feat in feat_element:
            new_feat_element.append(str(a_feat) + '_' + a_label)
        new_features.append(feat_element + new_feat_element)
    return new_features


def add_bigram_features_test(features, a_label):
    '''
    Add previous label for features of one word
    This is used as a replacement of add_bigram_features procedure
    for test, where only one previous label at a time
    is available.
    '''
    new_features = []
    for a_feat in features:
        new_features.append(str(a_feat) + '_' + a_label)
    return features + new_features
