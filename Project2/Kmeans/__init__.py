# -*- coding: utf-8 -*-
from .processing import *
from .helpers import *
import pandas as pd
import numpy as np

def main(input_path, format_path, k):
    """Predicts the Netflix ratings using a modified version of Kmeans
    
    More precisely, it preprocesses the data to make it compatible with kmeans, 
    applies Kmeans (a version that ignores nans) with k clusters to it 
    then finally postprocesses the result to give predictions the desired format.

    Args:
        input_path: Path to the samples
        format_path: Path to the submission format file
        k: The amount of clusters
        
    Returns:
        np.array: The prediction
    """
    #useful constants
    max_iters = 100
    threshold = 1e-6
    #preprocessing
    data=preprocessing(input_path)
    #kmeans
    print('Kmeans for k=',k,':')
    assignments, mu, _=kmeans(data, k, max_iters, threshold)
    #generating a prediction using the cluster of similar users
    data=cluster_agg(assignments, mu, k, data)
    #postprocessing
    data=postprocessing(data, format_path)
    return data

if __name__ == "__main__":
    main("../data/data_train.csv", "../data/sampleSubmission.csv")