# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
# importing the models
import Kmeans
import NN
import ALS


def median_vote(predictions):
    """For each entry of the prediction, keeps the median.

    More precisely, the median is taken if the amount of entries is odd, and the rounded down version of it is taken if
    the amount of entries is odd.

    Args:
        predictions: The array of predictions

    Returns:
        np.array: The aggregated prediction
    """
    # computing the median
    median = np.median(predictions, axis=0)
    # making sure the result is an int (not the case if the amount of predictions is even)
    bounded_median = np.floor(median)
    return bounded_median


def mode_vote(predictions):
    """For each entry of the prediction, keeps the mode.

    Args:
        predictions: The array of predictions

    Returns:
        np.array: The aggregated prediction
    """
    # computing the mode
    return stats.mode(predictions, axis=0)[0]


def vote(voting_f):
    """Computes the prediction given by all the models and aggregates them using the given function.

    Args:
        voting_f: The aggregating function

    Returns:
        np.array: The aggregated prediction
    """
    #useful constants
    submission_path='submission.csv'
    training_path = "data/data_train.csv"
    format_path = "data/sampleSubmission.csv"
    #Loading the data
    print("Loading datasets")
    try:
        input_ = pd.read_csv(training_path)
        format_ = pd.read_csv(format_path)
    except FileNotFoundError:
        print("Impossible to load training or format files, "
              "please double check")
        return pd.DataFrame([])
    #computing the prediction of the ALS algorithm
    predictions=ALS.main(input_.copy(), format_.copy())
    #computing multiple predictions of the kmeans algorithm
    for k in [5,6,7]:
        predictions=predictions.merge(Kmeans.main(input_.copy(), format_.copy(), k),on='Id')
    #computing the prediction of the NN algorithm
    predictions=predictions.merge(NN.main(input_.copy(), format_.copy()),on='Id')
    #setting 'Id' as the index of the aggregation of predictions
    predictions.set_index('Id', inplace=True)
    #finding the best prediction through the voting function
    print('Voting...')
    predictions['Prediction']=voting_f(predictions.T)
    #exporting the final prediction using the submission path
    print('Exporting the final prediction...')
    predictions[['Prediction']].to_csv(submission_path)
    print('Done!')