# -*- coding: utf-8 -*-
from processing import *
from helpers import *
import pandas as pd
import numpy as np

def main(k):
    """Predicts the Netflix ratings using a modified version of Kmeans
    
    More precisely, it preprocesses the data to make it compatible with kmeans, 
    applies Kmeans (a version that ignores nans) with k clusters to it 
    then finally postprocesses the result to give predictions the desired format.

    Args:
        k: The amount of clusters
        
    Returns:
        np.array: The prediction
    """
    #useful constants
    input_path='../Data/data_train.csv'
    format_path='../Data/sampleSubmission.csv'
    max_iters = 100
    threshold = 1e-6
    #optimal k computed empirically
    k_opt=k#6
    #preprocessing
    data=preprocessing(input_path)
    #kmeans
    assignments, mu, _=kmeans(data, k_opt, max_iters, threshold)
    #generating a prediction using the clusters of similar users
    mu_rounded=np.round(mu)
    for i in range(k_opt):
        data[assignments==i]=np.where(np.isnan(data[assignments==i]),mu_rounded[i],data[assignments==i])
    #postprocessing
    data=postprocessing(data, format_path)
    return data