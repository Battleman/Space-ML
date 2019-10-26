# Useful starting lines
import numpy as np

from preprocessing import preprocessing
from features_engineering import augment
from ml_methods import (least_squares,
                        ridge_regression,
                        reg_logistic_regression_GD,
                        reg_logistic_regression_SGD,
                        logistic_regression_SGD)
from proj1_helpers import load_csv_data, create_csv_submission, predict_labels

CACHE = False

NUM_SETS = 3

COMBINED_DEGREES = [3, 3, 2]
SIMPLE_DEGREES = [3, 4, 4]
TAN_HYP_DEGREES = [4, 4, 4]
INVERSE_LOG_DEGREES = [4, 4, 4]
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
y, tX_train, _ = load_csv_data(DATA_TRAIN_PATH)
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

#########
# Preprocess
#########
print("Preprocessing")
(XS_TRAIN, MASKS_TRAIN) = preprocessing(tX_train)
(XS_TEST, MASKS_TEST) = preprocessing(tX_test)

# placeholder for submission
y_submission = np.zeros(tX_test.shape[0])

# compute for each subset of PRI_JET_NUM
for i in range(NUM_SETS):
    print("#####Working on jet_num={}#####".format(i))
    y_correspond = y[MASKS_TRAIN[i]]

    # features engineering train set
    print("Augmenting training set")
    x_train_aug_fname = ("cache/x_train_augmented_jet{}_{}dim.np"
                         .format(i, COMBINED_DEGREES[i]))
    try:
        with open(x_train_aug_fname, "rb") as f:
            print("Augmented training set cached found")
            x_train_aug = np.load(f)
    except FileNotFoundError:
        # not existing, recomputing
        print("No augmented training set cached, recomputing")
        x_train_aug = augment(XS_TRAIN[i],
                              COMBINED_DEGREES[i],
                              SIMPLE_DEGREES[i],
                              TAN_HYP_DEGREES[i],
                              INVERSE_LOG_DEGREES[i])
        if CACHE:
            with open(x_train_aug_fname, "wb") as f:
                np.save(f, x_train_aug)

    # Compute weights
    # TODO change function to get weights
    print("Computing optimal weights")
    # w, _ = least_squares(y_correspond, x_train_aug)
    w, _ = ridge_regression(y_correspond, x_train_aug, 1e-5)
    # w, _ = logistic_regression_SGD(y_correspond, x_train_aug, [0.0]*x_train_aug.shape[1], 1, 100000, 1e-3)
    print(w, np.unique(w))
    del x_train_aug

    # features engineering test set
    print("Augmenting testing set")
    x_test_aug_fname = "cache/x_test_augmented_jet{}_{}dim.np".format(
        i, COMBINED_DEGREES[i])
    try:
        with open(x_test_aug_fname, "rb") as f:
            x_test_aug = np.load(f)
    except FileNotFoundError:
        # not existing, recomputing
        x_test_aug = augment(XS_TEST[i],
                             COMBINED_DEGREES[i],
                             SIMPLE_DEGREES[i],
                             TAN_HYP_DEGREES[i],
                             INVERSE_LOG_DEGREES[i])
        if CACHE:
            with open(x_test_aug_fname, "wb") as f:
                np.save(f, x_test_aug)

    # compute predictions and store
    print("Predicting labels for subset")
    y_submission[MASKS_TEST[i]] = predict_labels(w, x_test_aug)
    del x_test_aug
    del w
# all predictions completed, create CSV
print("Creating submission")
create_csv_submission(ids_test, y_submission, OUTPUT_PATH)
