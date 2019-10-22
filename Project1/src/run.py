# Useful starting lines
import numpy as np

from preprocessing import preprocessing
from features_engineering import augment
from ml_methods import (least_squares,
                        ridge_regression,
                        reg_logistic_regression_GD,
                        reg_logistic_regression_SGD)
from proj1_helpers import load_csv_data, create_csv_submission, predict_labels

poly_deg = 3
#########
# File paths
#########
DATA_TRAIN_PATH = '../data/train.csv'
DATA_TEST_PATH = '../data/test.csv'
OUTPUT_PATH = '../predictions.csv'

#########
# Load CSV
#########
print("Loading CSV")
y, tX_train, ids = load_csv_data(DATA_TRAIN_PATH)
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

#########
# Preprocess
#########
print("Preprocessing")
(xs_train, masks_train) = preprocessing(tX_train)
(xs_test, masks_test) = preprocessing(tX_test)

# placeholder for submission
y_submission = np.zeros(tX_test.shape[0])

# compute for each subset of PRI_JET_NUM
jet_num = 0
for x_train, mask_train, x_test, mask_test in zip(xs_train, masks_train, xs_test, masks_test):
    print("Working on jet_num={}".format(jet_num))
    y_correspond = y[mask_train]

    # features engineering train set
    print("Augmenting training set")
    x_train_aug_fname = "cache/x_train_augmented_jet{}_{}dim.np".format(jet_num, poly_deg)
    try:
        with open(x_train_aug_fname, "rb") as f:
            print("Augmented training set cached found")
            x_train_aug = np.load(f)
    except FileNotFoundError:
        # not existing, recomputing
        print("No augmented training set cached, recomputing")
        x_train_aug = augment(x_train, poly_deg)
        with open(x_train_aug_fname, "wb") as f:
            np.save(f, x_train_aug)
    del x_train

    # Compute weights
    # TODO change function to get weights
    print("Computing optimal weights")
    # w, _ = least_squares(y_correspond, x_train_aug)
    w, _ = ridge_regression(y_correspond, x_train_aug, 0.2)
    del x_train_aug

    # features engineering test set
    print("Augmenting testing set")
    x_test_aug_fname = "cache/x_test_augmented_jet{}_{}dim.np".format(jet_num, poly_deg)
    try:
        with open(x_test_aug_fname, "rb") as f:
            x_test_aug = np.load(f)
    except FileNotFoundError:
        # not existing, recomputing
        x_test_aug = augment(x_test, poly_deg)
        with open(x_test_aug_fname, "wb") as f:
            np.save(f, x_test_aug)
    del x_test

    # compute predictions and store
    print("Predicting labels for subset")
    y_submission[mask_test] = predict_labels(w, x_test_aug)
    del x_test_aug
    del w
    jet_num += 1
# all predictions completed, create CSV
print("Creating submission")
create_csv_submission(ids_test, y_submission, OUTPUT_PATH)
