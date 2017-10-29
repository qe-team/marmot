#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_wmt16_task1
~~~~~~~~~~~~~~~~~~~~

Evaluation script for Task 1 of the WMT Quality Estimation challenge.

:copyright: (c) 2016 by Fabio Kepler
:licence: MIT

Usage:
  evaluate_wmt16_task1 [options] REFERENCE_FILE SUBMISSION_FILE...
  evaluate_wmt16_task1 (-h | --help | --version)

Arguments:
  REFERENCE_FILE        path to a reference file in either a tab-separated format
                        <METHOD NAME> <SEGMENT NUMBER> <SEGMENT SCORE> <SEGMENT RANK>
                        or with one HTER score per line;
                        format will be detected based on the first line
  SUBMISSION_FILE...    list of submission files with the same format options as REFERENCE_FILE

Options:
  -s --scale FACTOR     FACTOR by which to scale (multiply) input scores
  -v --verbose          log debug messages
  -q --quiet            log only warning and error messages

Other:
  -h --help             show this help message and exit
  --version             show version and exit
"""

import logging

import numpy as np
import sklearn.metrics as sk
from docopt import docopt
from scipy.stats.stats import pearsonr, spearmanr, rankdata

__prog__ = "evaluate_wmt16_task1"
__title__ = 'Evaluate WMT2016 Quality Estimation Task 1'
__summary__ = 'Evaluation script for Task 1 of the WMT Quality Estimation challenge.'
__uri__ = 'https://gist.github.com/kepler/6043a41ed8f3ed0be1e68c5942b99734'
__version__ = '0.0.1'
__author__ = 'Fabio Kepler'
__email__ = 'fabio@kepler.pro.br'
__license__ = 'MIT'
__copyright__ = 'Copyright 2016 Fabio Kepler'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def delta_average(y_true, y_rank):
    """
    Calculate the DeltaAvg score.

    References: ?

    :param y_true: array of reference score (not rank) of each segment.
    :param y_rank: array of rank of each segment.
    :return: the absolute delta average score.
    """
    sorted_ranked_indexes = np.argsort(y_rank)
    y_length = len(sorted_ranked_indexes)

    delta_avg = 0
    max_quantiles = y_length // 2
    set_value = np.sum(y_true[sorted_ranked_indexes[np.arange(y_length)]]) / y_length
    quantile_values = {
        head: np.sum(y_true[sorted_ranked_indexes[np.arange(head)]]) / head for head in range(2, y_length)
        }  # cache values, since there are many that are repeatedly computed between various quantiles
    for quantiles in range(2, max_quantiles + 1):  # current number of quantiles
        quantile_length = y_length // quantiles
        quantile_sum = 0
        for head in np.arange(quantile_length, quantiles * quantile_length, quantile_length):
            quantile_sum += quantile_values[head]
        delta_avg += quantile_sum / (quantiles - 1) - set_value

    if max_quantiles > 1:
        delta_avg /= (max_quantiles - 1)
    else:
        delta_avg = 0
    return abs(delta_avg)


def parse_submission(file_name):
    """
    <METHOD NAME>\t<SEGMENT NUMBER>\t<SEGMENT SCORE>\t<SEGMENT RANK>
    """
    with open(file_name) as f:
        sentences = [line.strip().split('\t') for line in f]
    method = set(map(lambda x: x[0], sentences))
    if len(method) > 1:
        logger.error('There is more than one method name in file "{}": {}'.format(file_name, method))
        return None, None
    method = list(method)[0]
    segments = np.asarray(list(map(lambda x: x[1:], sentences)), dtype=float)
    if segments[:, 0].max() != segments.shape[0]:
        logger.error('Wrong number of segments in file "{}": found {}, expected {}.'.format(file_name, segments.shape[0], segments[:, 0].max()))
        return None, None
    return method, segments


def read_hter(file_name):
    with open(file_name) as f:
        scores = np.array([line.strip() for line in f], dtype='float')
    method = file_name
    segments = np.vstack((np.arange(1, scores.shape[0] + 1),
                          scores,
                          rankdata(scores, method='ordinal'))).T
    return method, segments


def read_file(file_name):
    with open(file_name) as f:
        if '\t' in f.readline().strip():
            return parse_submission(file_name)
        else:
            return read_hter(file_name)


def run(arguments):
    reference_file = arguments['REFERENCE_FILE']
    submission_files = arguments['SUBMISSION_FILE']

    reference_method, reference_segments = read_file(reference_file)
    if arguments['--scale']:
        reference_segments[:, 1] *= float(arguments['--scale'])

    scoring_values = []
    ranking_values = []
    for submission in submission_files:
        submission_method, submission_segments = read_file(submission)
        if arguments['--scale']:
            submission_segments[:, 1] *= float(arguments['--scale'])

        if submission_segments[:, 1].any():
            pearson = pearsonr(reference_segments[:, 1], submission_segments[:, 1])[0]  # keep only main value
            mae = sk.regression.mean_absolute_error(reference_segments[:, 1], submission_segments[:, 1])
            rmse = np.sqrt(sk.regression.mean_squared_error(reference_segments[:, 1], submission_segments[:, 1]))
            scoring_values.append((submission_method, pearson, mae, rmse))
        if submission_segments[:, 2].any():
            spearman = spearmanr(reference_segments[:, 2], submission_segments[:, 2])[0]  # keep only main value
            delta_avg = delta_average(reference_segments[:, 1], submission_segments[:, 2])  # DeltaAvg needs reference scores instead of rank
            ranking_values.append((submission_method, spearman, delta_avg))

    scoring = np.array(scoring_values, dtype=[('Method', 'object'), ('Pearson r', float), ('MAE', float), ('RMSE', float)])
    logger.info('Scoring results:')
    logger.info('{:20} {:20} {:20} {:20}'.format('Method', 'Pearson r', 'MAE', 'RMSE'))
    for submission in np.sort(scoring, order=['Pearson r', 'MAE', 'RMSE']):
        logger.info('{:20s} {:<20.10} {:<20.10} {:<20.10}'.format(*submission))

    ranking = np.array(ranking_values, dtype=[('Method', 'object'), ('Spearman rho', float), ('DeltaAvg', float)])
    logger.info('Ranking results:')
    logger.info('{:20} {:20} {:20}'.format('Method', 'Spearman rho', 'DeltaAvg'))
    for submission in np.sort(ranking, order=['Spearman rho', 'DeltaAvg']):
        logger.info('{:20} {:<20.10} {:<20.10}'.format(*submission))


if __name__ == '__main__':
    options = docopt(__doc__, argv=None, help=True, version=__version__, options_first=False)

    if options['--verbose']:
        logger.setLevel(level='DEBUG')
    elif options['--quiet']:
        logger.setLevel(level='WARNING')

    run(options)
    
