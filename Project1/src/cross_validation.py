import numpy as np
from implementations import ridge_regression
from proj1_helpers import predict_labels


def k_fold_splits(y, x, num_folds):
    rows = len(y)

    # build k_indices
    interval = rows // num_folds
    np.random.seed(seed=1)
    rand_indices = np.random.permutation(rows)
    k_indices = np.array([rand_indices[k * interval: (k + 1) * interval]
                 for k in range(num_folds)])

    # get k'th subgroup in validation, others in train
    kfold_return = []
    for k in range(num_folds):
        val_ind = k_indices[k]
        tr_indices = k_indices[~(np.arange(k_indices.shape[0]) == k)]
        tr_indices = tr_indices.reshape(-1)
        kfold_return.append([np.array(x)[tr_indices.astype(int)], np.array(x)[val_ind.astype(int)],y[tr_indices], y[val_ind]])
    return kfold_return


def cross_validation_ridge(y_train, x_train, num_folds, lambda_, seed=1):
    np.random.seed(seed)
    scores = []
    for x_train_sub, x_val_sub, y_train_sub, y_val_sub in k_fold_splits(y_train, x_train, num_folds):
        w, _ = ridge_regression(y_train_sub, x_train_sub, lambda_)
        y_val_predict = predict_labels(w, x_val_sub)
        score = np.mean(y_val_predict == y_val_sub)
        scores.append(score)
    return np.array(scores)
