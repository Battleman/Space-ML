# -*- coding: utf-8 -*-
try:
    from processing import postprocessing, preprocessing
    from helpers import cluster_agg, kmeans
except ImportError:
    from .processing import postprocessing, preprocessing
    from .helpers import cluster_agg, kmeans
import pandas as pd


def main(input_, format_, k=6, rounded=True):
    """Predicts the Netflix ratings using a modified version of Kmeans

    More precisely, it preprocesses the data to make it compatible with kmeans,
    applies Kmeans (a version that ignores nans) with k clusters to it
    then finally postprocesses the result to give predictions the desired
    format.

    Args:
        input_: The samples
        format_: Submission format file
         k: The amount of clusters
        rounded: Whether to round the predictions or not

    """
    # useful constants
    max_iters = 100
    threshold = 1e-6
    # preprocessing
    data = preprocessing(input_)
    # kmeans
    print('Kmeans for k=', k, ':')
    assignments, mu, _ = kmeans(data, k, max_iters, threshold)
    # generating a prediction using the cluster of similar users
    data = cluster_agg(assignments, mu, k, data, rounded=rounded)
    # postprocessing
    data = postprocessing(data, format_)
    return data


if __name__ == "__main__":
    main(pd.read_csv("../data/data_train.csv"),
         pd.read_csv("../data/sampleSubmission.csv"), 6)
