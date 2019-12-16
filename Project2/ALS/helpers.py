# -*- coding: utf-8 -*-
"""some functions for help."""

from itertools import groupby

import numpy as np
import scipy.sparse as sp
import pickle as pkl


def load_data(path_dataset):
    """Load data in text format, one rating per line."""
    with open(path_dataset, "r") as f:
        data = f.read().splitlines()[1:]
    return _preprocess_data(data)


def _preprocess_data(data):
    """Preprocessing the text data, conversion to numerical array format."""
    def deal_line(line):
        pos, rating = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row), int(col), float(rating)

    def statistics(data):
        row = set([line[0] for line in data])
        col = set([line[1] for line in data])
        return min(row), max(row), min(col), max(col)

    # parse each line
    data = [deal_line(line) for line in data]

    # do statistics on the dataset.
    min_row, max_row, min_col, max_col = statistics(data)

    # build rating matrix.
    ratings = sp.lil_matrix((max_row, max_col))
    for row, col, rating in data:
        ratings[row - 1, col - 1] = rating
    return ratings


def deserialize_costs(costs_filename):
    try:
        with open(costs_filename, "rb") as f:
            print("Successfully retrieved cached lambdas optimization")
            costs = pkl.load(f)
    except FileNotFoundError:
        print("Unable to retrieve cached lambdas optimization, " +
              "starting from scratch")
        costs = {}
    return costs


def group_by(data, index):
    """group list of list by a specific index."""
    sorted_data = sorted(data, key=lambda x: x[index])
    groupby_data = groupby(sorted_data, lambda x: x[index])
    return groupby_data


def build_index_groups(train):
    """build groups for nnz rows and cols."""
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))

    grouped_nz_train_byrow = group_by(nz_train, index=0)
    nz_row_colindices = [(g, np.array([v[1] for v in value]))
                         for g, value in grouped_nz_train_byrow]

    grouped_nz_train_bycol = group_by(nz_train, index=1)
    nz_col_rowindices = [(g, np.array([v[0] for v in value]))
                         for g, value in grouped_nz_train_bycol]
    return nz_train, nz_row_colindices, nz_col_rowindices


def update_user_feature(
        train, item_features, lambda_user,
        nnz_items_per_user, nz_user_itemindices):
    """update user feature matrix."""
    """the best lambda is assumed to be nnz_items_per_user[user] * lambda_user"""
    num_user = nnz_items_per_user.shape[0]
    num_feature = item_features.shape[0]
    lambda_I = lambda_user * sp.eye(num_feature)
    updated_user_features = np.zeros((num_feature, num_user))

    for user, items in nz_user_itemindices:
        # extract the columns corresponding to the prediction for given item
        M = item_features[:, items]

        # update column row of user features
        V = M @ train[items, user]
        A = M @ M.T + nnz_items_per_user[user] * lambda_I
        X = np.linalg.solve(A, V)
        updated_user_features[:, user] = np.copy(X.T)
    return updated_user_features


def update_item_feature(
        train, user_features, lambda_item,
        nnz_users_per_item, nz_item_userindices):
    """update item feature matrix."""
    """the best lambda is assumed to be nnz_items_per_item[item] * lambda_item"""
    num_item = nnz_users_per_item.shape[0]
    num_feature = user_features.shape[0]
    lambda_I = lambda_item * sp.eye(num_feature)
    updated_item_features = np.zeros((num_feature, num_item))

    for item, users in nz_item_userindices:
        # extract the columns corresponding to the prediction for given user
        M = user_features[:, users]
        V = M @ train[item, users].T
        A = M @ M.T + nnz_users_per_item[item] * lambda_I
        X = np.linalg.solve(A, V)
        updated_item_features[:, item] = np.copy(X.T)
    return updated_item_features


def split_data(ratings, p_test=0.1):
    """split the ratings to training data and test data.
    Args:
        min_num_ratings:
            all users and items we keep must have at least min_num_ratings per user and per item.
    """
    # set seed
    np.random.seed(988)

    # init
    num_rows, num_cols = ratings.shape
    train = sp.lil_matrix((num_rows, num_cols))
    test = sp.lil_matrix((num_rows, num_cols))

    print("the shape of original ratings. (# of row, # of col): {}".format(
        ratings.shape))

    nz_items, nz_users = ratings.nonzero()

    # split the data
    set_nz_users = set(nz_users)
    for i, user in enumerate(set_nz_users):
        print("Splitting progression: {}%".format(
            100*(i+1)/len(set_nz_users)), end="\r")
        # randomly select a subset of ratings
        row, col = ratings[:, user].nonzero()
        selects = np.random.choice(row, size=int(len(row) * p_test))
        residual = list(set(row) - set(selects))

        # add to train set
        for r in residual:
            train[r, user] = ratings[r, user]

        # add to test set
        for s in selects:
            test[s, user] = ratings[s, user]
    print("")
    print("Total number of nonzero elements in original data:{v:,}".format(
        v=ratings.nnz))
    print("Total number of nonzero elements in train data:{v:,}".format(
        v=train.nnz))
    print(
        "Total number of nonzero elements in test data:{v:,}".format(v=test.nnz))
    return train, test


def init_MF(train, num_features):
    """Init the parameter for matrix factorization."""
    num_item, num_user = train.get_shape()
    user_features = np.random.rand(num_features, num_user)
    item_features = np.random.rand(num_features, num_item)

    # start by item features.
    item_nnz = train.getnnz(axis=1)
    item_sum = train.sum(axis=1)
    for ind in range(num_item):
        item_features[0, ind] = item_sum[ind, 0] / item_nnz[ind]
    return user_features, item_features


def compute_error(data, user_features, item_features, nz):
    """compute the loss (MSE) of the prediction of nonzero elements."""
    mse = 0
    for row, col in nz:
        item_info = item_features[:, row]
        user_info = user_features[:, col]
        mse += (data[row, col] - user_info.T.dot(item_info)) ** 2
    return np.sqrt(mse / len(nz))


def ALS(train, test=None, lambda_user=0.1, lambda_item=0.7,
        num_features=40, max_steps=30):
    """Alternating Least Squares (ALS) algorithm."""
    # define parameters
    stop_criterion = 1e-4
    change = 1
    error_list = [0, 0]

    # set seed
    np.random.seed(988)

    # init ALS
    user_features, item_features = init_MF(train, num_features)

    # get the number of non-zero ratings for each user and item
    nnz_items_per_user, nnz_users_per_item = train.getnnz(
        axis=0), train.getnnz(axis=1)

    # group the indices by row or column index
    nz_train, nz_item_userindices, nz_user_itemindices = build_index_groups(
        train)

    # run ALS
    print("Using lambda_user={:.5f}, lambda_item={:.5f}\n"
          "start the ALS algorithm...".format(lambda_user, lambda_item))
    step = 0
    while step < max_steps:
        if change < stop_criterion:
            print("")
            print("Quitting because converged")
            break
        # update user feature & item feature
        user_features = update_user_feature(train,
                                            item_features,
                                            lambda_user,
                                            nnz_items_per_user,
                                            nz_user_itemindices)
        item_features = update_item_feature(train,
                                            user_features,
                                            lambda_item,
                                            nnz_users_per_item,
                                            nz_item_userindices)

        error = compute_error(train, user_features, item_features, nz_train)
        print("Step {}, RMSE on training set: {}."
              .format(step, error), end="\r")
        error_list.append(error)
        change = np.fabs(error_list[-1] - error_list[-2])
        step += 1
    else:
        # didn't break out of loop, so max step reached
        print("")
        print("CONVERGENCE TOO SLOW, INTERRUPTED")

    # if necessary, evaluate the test error
    rmse = None
    if test is not None:
        nnz_row, nnz_col = test.nonzero()
        nnz_test = list(zip(nnz_row, nnz_col))
        rmse = compute_error(test, user_features, item_features, nnz_test)
        print("test RMSE after running ALS: {v}.".format(v=rmse))
    return (user_features, item_features), rmse
