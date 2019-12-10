import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import matplotlib.pyplot as plt
from helpers import (build_index_groups, load_data, preprocess_data, split_data, 
                     ALS, update_item_feature, update_user_feature, compute_mf, 
                     compute_final, compute_error)
from plot import plot_raw_data, plot_train_test_data
import numpy as np
import plotly.graph_objects as go
import pickle as pkl

def main_ALS(path_dataset, path_submission, min_num_costs=130):
    #load data and split
    print("Loading datasets")
    ratings = load_data(path_dataset)
    final = load_data(path_submission)
    num_items_per_user = np.array((ratings != 0).sum(axis=0)).flatten()
    num_users_per_item = np.array((ratings != 0).sum(axis=1).T).flatten()
    print("Spliting train/test")
    _, train, test = split_data(ratings, 
                                num_items_per_user, 
                                num_users_per_item, 
                                min_num_ratings=0, 
                                p_test=0.1)
    
    #load pre-saved costs
    print("Trying to retrieve cached lambdas optimization")
    costs_filename = "cache/als_costs.pkl"
    try:
        with open(costs_filename, "rb") as f:
            print("Successfully retrieved cached lambdas optimization")
            costs = pkl.load(f)
    except FileNotFoundError:
        print("Unable to retrieve cached lambdas optimization, starting from scratch")
        costs = {}
    #if necessary, compute some more
    while len(costs) < min_num_costs:
        print("Not enough samples ({}/{}), computing...".format(len(costs), min_num_costs))
        u = np.random.sample()
        i = np.random.sample()
        if (u,i) not in costs:
            costs[(u,i)] = ALS(train, test, u, i)
            with open(costs_filename, "wb") as f:
                pkl.dump(costs, f)

    #try to retrieve best matrix factorization
    print("Trying to retrieve cached optimal matrix factorization")
    factorized_filename = "cache/factorized.pkl"
    try:
        with open(factorized_filename, "rb") as f:
            print("Successfully retrieved cached optimal matrix factorization")
            factorized = pkl.load(f)
    except FileNotFoundError:
        #if failed, recompute and cache
        print("Unable to retrieve cached optimal matrix factorization, computing")
        (min_ulambda, min_ilambda,), _ = min(costs.items(), key=lambda x: x[1])
        factorized = compute_mf(train, min_ulambda, min_ilambda, 40)
        with open(factorized_filename, "wb") as f:
            print("Caching optimal matrix factorization")
            factorized = pkl.dump(factorized, f)
    ufeats, ifeats = factorized
    
    
    nnz_row, nnz_col = final.nonzero()
    nnz_final = list(zip(nnz_row, nnz_col))
    ret = []
    i=1
    for row, col in nnz_final:
        print("Emitting predictions {}/{}".format(i, len(nnz_final)), end="\r")
        item_info = ifeats[:, row]
        user_info = ufeats[:, col]
        ret.append("r{}_c{},{}".format(row+1, 
                                       col+1, 
                                       int(np.clip(np.round(user_info.T.dot(item_info)), 1, 5))))
        i += 1
    return ret
if __name__ == "__main__":
    print(main("../data/data_train.csv", "../data/ALS.csv"))