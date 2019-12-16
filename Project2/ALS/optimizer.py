# -*- coding: utf-8 -*-
"""Define methods related to optimization of hyperparameters."""

try:
    from .helpers import ALS, deserialize_costs, load_data, split_data
except (ModuleNotFoundError, ImportError):
    from helpers import ALS, deserialize_costs, load_data, split_data
import os
import sys
import numpy as np
import pickle as pkl


def optimizer_lambdas(min_num_costs, train, test, close_to_best=False, num_features=40):
    """Optimize the regularization hyperparameters

    Arguments:
        min_num_costs {int} -- Min number of samples of the hyperparameters
        space to have
        train {scipy sparse matrix} -- Training matrix
        test {scipy sparse matrix} -- Testing matrix

    Keyword Arguments:
        close_to_best {bool} -- Should the optimization try to stay close to
        the current best point (default: {False})

    """
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    costs_filename = CURRENT_DIR+"/cache/als_costs_{}.pkl".format(num_features)
    costs = deserialize_costs(costs_filename)
    while len(costs) < min_num_costs:
        print("Not enough samples ({}/{}), computing..."
              .format(len(costs), min_num_costs))
        best_params = (0,0)
        if len(costs) > 0:
            best_param = min(costs, key=lambda x: costs[x])

        while True:
            # looking for suitable parameters
            u = np.random.random_sample()
            i = np.random.random_sample()
            if close_to_best:
                # only keep points close to current best point
                dist_to_best = np.linalg.norm(
                    (np.array(best_param) - np.array([u, i])))
            else:
                # take any point
                dist_to_best = 0
            if dist_to_best < 0.2:
                if (u, i) not in costs:
                    print("")
                    print("FOUND!")
                    break
                else:
                    print("Parameters already computed, keep searching..."
                          .format(u, i),
                          end="\r")
            else:
                print("Point too far from current best, keep searching...",
                      end="\r")
        _, _, c = ALS(train, test, u, i, num_features=num_features)
        costs[(u, i)] = c
        with open(costs_filename, "wb") as f:
            pkl.dump(costs, f)
    return get_best_lambdas(num_features)

def get_best_lambdas(num_features):
    """Return the best paramters."""
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    costs_filename = CURRENT_DIR+"/cache/als_costs_{}.pkl".format(num_features)
    costs = deserialize_costs(costs_filename)
    if len(costs) == 0:
        return (None, None)
    return min(costs, key=lambda x: costs[x])


if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    path_dataset = CURRENT_DIR+"/../data/data_train.csv"
    ratings = load_data(path_dataset)
    tr, te = split_data(ratings, p_test=0.1)
    optimizer_lambdas(int(sys.argv[1]), tr, te, close_to_best=True)
