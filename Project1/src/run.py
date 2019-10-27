# Useful starting lines
import numpy as np
import yaml

from features_engineering import augment
from implementations import (least_squares, logistic_regression_SGD,
                        reg_logistic_regression_GD,
                        reg_logistic_regression_SGD, ridge_regression)
from preprocessing import preprocessing
from proj1_helpers import create_csv_submission, load_csv_data, predict_labels

with open("parameters.yaml") as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

COMBINED_DEGREES = params['COMBINED_DEGREES']
SIMPLE_DEGREES = params['SIMPLE_DEGREES']
TAN_HYP_DEGREES = params['TAN_HYP_DEGREES']
INVERSE_LOG_DEGREES = params['INVERSE_LOG_DEGREES']
ROOT_DEGREES = params['ROOT_DEGREES']
NUM_SETS = params['NUM_SETS']
DATA_TRAIN_PATH = params['DATA_TRAIN_PATH']
DATA_TEST_PATH = params['DATA_TEST_PATH']

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
