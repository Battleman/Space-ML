# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import keras
from keras.models import Model


def preprocessing(path):
    """ Loads dataset from path and preprocesses it for NN training.

    This also splits the data into a training set and a testing set
    to perform a cross validation during the NN training.

    Args:
        path: Path to the data

    """
    # Loading the training data
    data = pd.read_csv(path)
    # Transform data into 2 vectors UserID and MovieId
    data['userId'] = data['Id'].apply(
        lambda x: x.split('_')[0][1:]).astype('int')
    data['movieId'] = data['Id'].apply(
        lambda x: x.split('_')[1][1:]).astype('int')
    data.drop('Id', axis=1, inplace=True)

    # Calculate the number of unique users and movies in the dataset
    n_users = data['userId'].nunique()
    n_movies = data['movieId'].nunique()

    # Getting the desired input for the NN
    X = data[['userId', 'movieId']].values
    y = data['Prediction'].values

    # Splits data into 90% training and 10% testing set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42)

    X_train_array = [X_train[:, 0], X_train[:, 1]]
    X_test_array = [X_test[:, 0], X_test[:, 1]]

    return X_train_array, y_train, X_test_array, y_test, n_users, n_movies


def get_uniquevalues(data):
    """ Calculates the number of unique users and movies in the dataset

    Args:
        data: Dataframe of original data transformed into 2 user/movie columns

    """
    n_users = data['userId'].nunique()
    n_movies = data['movieId'].nunique()

    return n_users, n_movies


def predict(model, path):
    """Return predictions of the trained NN model.

    Loads the data given to predict from `path`, runs the trained NN `model`
    on this data and finally output the ratings predictions.

    Args:
        model: The trained NN model
        path: Path to the data given for prediction

    Returns:
        pandas dataframe -- dataframe of ids and corresponding ratings
    """
    # Loading the sample submission data
    sample_sumbission = pd.read_csv(path)
    # Transforming data to have the correct format for the NN
    sample_sumbission['userId'] = sample_sumbission['Id'].apply(
        lambda x: x.split('_')[0][1:]).astype('int')
    sample_sumbission['movieId'] = sample_sumbission['Id'].apply(
        lambda x: x.split('_')[1][1:]).astype('int')
    sample_sumbission.drop('Id', axis=1, inplace=True)

    # Get the desired data for prediction
    X_p = sample_sumbission[['userId', 'movieId']].values
    X_pred = [X_p[:, 0], X_p[:, 1]]

    # Predict the ratings using the trained NN
    predictions = model.predict(X_pred)
    # round predictions
    rounded = [round(x[0]) for x in predictions]

    # Transform the data into the desired format
    sample_sumbission['Prediction'] = rounded
    sample_sumbission['Id'] = 'r' + sample_sumbission['userId'].astype('str') + '_c' + sample_sumbission[
        'movieId'].astype('str')
    sample_sumbission.drop(columns=["userId", "movieId"], inplace=True)
    columnsTitles = ["Id", "Prediction"]
    sample_sumbission = sample_sumbission.reindex(columns=columnsTitles)
    # Safety check
    sample_sumbission = sample_sumbission.replace(6, 5)

    return sample_sumbission
