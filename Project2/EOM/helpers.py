# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
#importing the models
sys.path.insert(0,'../Kmeans')
import _init_ as kmi
sys.path.insert(0,'../NN')
import _init_ as nni
sys.path.insert(0,'../ALS')
import _init_ as alsi

def median_vote(predictions):
    """For each entry of the prediction, keeps the median.

    More precisely, the median is taken if the amount of entries is odd, and the rounded down version of it is taken if 
    the amount of entries is odd.

    Args:
        predictions: The array of predictions
        
    Returns:
        np.array: The aggregated prediction
    """
    #computing the median
    median=np.median(predictions,axis=0)
    #making sure the result is an int (not the case if the amount of predictions is even)
    bounded_median=np.floor(median)
    return bounded_median

def mode_vote(predictions):
    """For each entry of the prediction, keeps the mode.

    Args:
        predictions: The array of predictions
        
    Returns:
        np.array: The aggregated prediction
    """
    #computing the mode
    return stats.mode(predictions,axis=0)[0]

def vote(voting_f):
    """Computes the prediction given by all the models and aggregates them using the given function.

    Args:
        voting_f: The aggregating function
        
    Returns:
        np.array: The aggregated prediction
    """
    #useful constants
    submission_path='submission.csv'
    #initializing 'predictions' and 'labels' using the ALS prediction
    predictions=alsi.main().to_numpy()
    labels=predictions[:,0]
    predictions=predictions[:,1]
    #adding the NN prediction to the list of predictions
    predictions=np.vstack(predictions,nni.main().to_numpy()[:,1]
    #adding all credible Kmeans predictions
    for k in [2,3,6,7]:
        predictions=np.vstack((predictions,kmi.main(k).to_numpy()[:,1]))
    #finding the best prediction though the voting function
    pred=pd.DataFrame(np.vstack((labels,voting_f(predictions))).T)
    pred=pred.rename(columns={0: 'Id', 1:'Prediction'})
    #exporting the final prediction using the submission path
    pred.to_csv(submission_path)