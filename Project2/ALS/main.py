# -*- coding: utf-8 -*-
# File ALS/main.py
"""Predict ratings using ALS."""
import os
import pickle as pkl

import numpy as np
import pandas as pd

try:
    from .helpers import ALS, load_data, split_data
    from .optimizer import get_best_lambdas, optimizer_lambdas
except (ModuleNotFoundError, ImportError):
    from helpers import ALS, load_data, split_data
    from optimizer import get_best_lambdas, optimizer_lambdas


def main(path_dataset, format_path, rounded=True, num_features=40):
    """Trains ALS and returns predictions.

    Loads dataset from `path_dataset`, performs ALS and predicts entries
    from `format_path`.

    To find optimal hyperparameters, samples the space of hyperparameters
    at least `min_num_costs` times and takes the best set of parameters.

    Arguments:
        path_dataset {str} -- Path (relative or absolute) to training dataset
        format_path {str} -- Path to entries for which emitting predictions.

    Keyword Arguments:
        min_num_costs {int} -- Min num of samples from hyperparameters
        space to take  (default: {130})

    Returns:
        list -- List of predictions, tuples of format ("id", rating)

    """
    # load data and split
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    print("Loading datasets")
    try:
        ratings = load_data(path_dataset)
        final = load_data(format_path)
    except FileNotFoundError:
        print("Impossible to load training or format files, "
              "please double check")
        return pd.DataFrame([])

    # try to retrieve best matrix factorization
    print("Trying to retrieve cached optimal matrix factorization")
    factorized_filename = CURRENT_DIR+"/cache/factorized.pkl"
    try:
        with open(factorized_filename, "rb") as f:
            print("Successfully retrieved cached optimal matrix factorization")
            factorized = pkl.load(f)
    except FileNotFoundError:
        # if failed, recompute and cache
        print("Unable to retrieve cached optimal matrix "
              "factorization, computing")
        min_ulambda, min_ilambda = get_best_lambdas(num_features)
        if min_ilambda is None or min_ilambda is None:
            print("Spliting train/test")
            train, test = split_data(ratings, p_test=0.1)
            min_ulambda, min_ilambda = optimizer_lambdas(150, train, test)
        factorized, _ = ALS(ratings,
                            lambda_user=min_ulambda,
                            lambda_item=min_ilambda,
                            max_steps=100,
                            num_features=num_features)
        with open(factorized_filename, "wb") as f:
            print("Caching optimal matrix factorization")
            pkl.dump(factorized, f)
    ufeats, ifeats = factorized

    nnz_row, nnz_col = final.nonzero()
    nnz_final = list(zip(nnz_row, nnz_col))
    ret = []
    i = 1
    for row, col in nnz_final:
        print("Emitting predictions {}/{}".format(i, len(nnz_final)), end="\r")
        item_info = ifeats[:, row]
        user_info = ufeats[:, col]
        r = user_info.T.dot(item_info)
        ret.append(("r{}_c{}".format(row+1, col+1),
                    (int(np.clip(np.round(r), 1, 5)) if rounded else r)))
        i += 1
    print("")
    return pd.DataFrame(ret, columns=["Id", "Prediction"])


if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    df = main(CURRENT_DIR+"/../data/data_train.csv",
              CURRENT_DIR+"/../data/sampleSubmission.csv")
    df.to_csv(CURRENT_DIR+"/cache/submissionALS.csv", index=False)
