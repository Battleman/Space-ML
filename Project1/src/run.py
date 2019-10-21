# Useful starting lines
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
%load_ext autoreload
%autoreload 2

from preprocessing import preprocessing
from features_engineering import augment
from ml_methods import *
from proj1_helpers import *
import pickle as pkl

#########
# File paths
#########
DATA_TRAIN_PATH = '../data/train.csv' 
DATA_TEST_PATH = '../data/test.csv'
OUTPUT_PATH = '../predictions.csv'

#########
# Load CSV
#########
y, tX_train, ids = load_csv_data(DATA_TRAIN_PATH)
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

#########
# Preprocess
#########
(xs_train, masks_train) = preprocessing(tX_train)
(xs_test, masks_test) = preprocessing(tX_test)

# placeholder for submission
y_submission = np.zeros(tX_test.shape[0])

# compute for each subset of PRI_JET_NUM
for x_train, mask_train, x_test, mask_test in zip(xs_train, masks_train, xs_test, masks_test):
    y_correspond = y[mask_train]
    #features engineering train
    x_train_aug = augment(x_train, 3)
    del x_train
    
    #Compute weights
    #TODO change function to get weights
    w,_ = least_squares(y_correspond, x_train_aug)
    del x_train_aug
    # features engineering test
    x_test_aug = augment(x_test, 3)
    del x_test
    #compute predictions and store
    y_submission[mask_test] = predict_labels(w, x_test_aug)
    del x_test_aug
    del w
# all predictions completed, create CSV
create_csv_submission(ids_test, y_submission, OUTPUT_PATH)