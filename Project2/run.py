# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy import stats
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures
import pickle as pkl
#importing the models
import Kmeans
import ALS
import NN
import Surprize


def augmentation(concat):
    '''Splits concat into X and y (if y is contained in it) and then applies a polynomial 3 feature augmentation to X
    Args:
        concat: The concatenation of X and y (or just X)
    Returns:
        (pandas.DataFrame,pandas.DataFrame): X augmented and y (None if there's no y)
    '''
    poly = PolynomialFeatures(3)
    X = concat.loc[:,~(concat.columns == "y")]
    print("Augmenting {} columns".format(len(X.columns)))
    if "y" in concat.columns:
        y = concat.loc[:, "y"]
    else:
        y = None
    X = pd.DataFrame(poly.fit_transform(X), columns=poly.get_feature_names(), index=concat.index)
    return X, y



#useful constants
submission_path='submission.csv'
training_path = "data/data_train.csv"
format_path = "data/sampleSubmission.csv"

#loading the data
print("Loading datasets")
try:
    input_ = pd.read_csv(training_path)
    format_ = pd.read_csv(format_path)
    with open('ridge_coefs.pkl','rb') as f:
        ridge_coefs=pkl.load(f)
except FileNotFoundError as e:
    print("Impossible to load training, format or hyperparameter files, "
          "please double check")
    raise e
    
#training the Surprize model
predictions_surprize_final = Surprize.main(input_.copy(), format_.copy(), cache_name="final")

#training the ALS model
predictions_als_final=ALS.main(input_.copy(), format_.copy(), cache_name="final")

#training the Kmeans variant model
predictions_kmeans_final = Kmeans.main(input_.copy(), format_.copy(), rounded=False)

#training the NN model
predictions_nn_final = NN.main(input_.copy(), format_.copy())

#aggregating all of our predictions into a single prediction matrix
concat_final = pd.concat([predictions_als_final, 
                predictions_kmeans_final, 
                predictions_nn_final, 
                predictions_surprize_final], axis=1, sort=False)
#freeing-up space
del predictions_surprize_final
del predictions_als_final
del predictions_kmeans_final
del predictions_nn_final
#applying a feature augmentation to our prediction matrix
concat_aug_final, _ = augmentation(concat_final)
del concat_final
#applying ridge regression to the prediction matrix to get the final prediction
#Optimal hyperparameter obtained through cross-validation
concat_aug_final["Prediction"] = 1.1945456804726544
predictor_coefficients=ridge_coefs
for i in range(len(concat_aug_final)):
    if col != "Prediction":
        concat_aug_final["Prediction"] += concat_aug_final.loc[:, i]*predictor_coefficients.get(i,0)
concat_aug_final["Prediction"] = concat_aug_final["Prediction"].apply(lambda x: int(np.clip(np.round(x),1,5)))
concat_aug_final.index.name = "Id"
#saving the final prediction
concat_aug_final.to_csv(submission_path, columns=["Prediction"])
