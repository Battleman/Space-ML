# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split


def preprocessing(data):
    """ Preprocesse the data used for the NN training.

    This transforms the data into 2 vectors one with userId values
    and one with movieId values.

    Args:
        data: The original training data

    """
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
    # X_train, X_test, y_train, y_test = train_test_split(
    #    X, y, test_size=0.1, random_state=42)

    X_all = [X[:, 0], X[:, 1]]
    # X_test_array = [X_test[:, 0], X_test[:, 1]]

    return X_all, y, n_users, n_movies


def get_uniquevalues(data):
    """ Calculate the number of unique users and movies in the dataset

    Args:
        data: Dataframe of original data transformed into 2 user/movie columns

    """
    n_users = data['userId'].nunique()
    n_movies = data['movieId'].nunique()

    return n_users, n_movies


def predict(model, format_):
    """Return predictions of the trained NN model.

    Runs the trained NN `model` on the data given to predict and
    finally output the ratings predictions.

    Args:
        model: The trained NN model
        format_: The data given for prediction

    Returns:
        pandas dataframe -- dataframe of ratings with their Ids as index
    """
    # Transforming data to have the correct format for the NN
    format_['userId'] = format_['Id'].apply(
        lambda x: x.split('_')[0][1:]).astype('int')
    format_['movieId'] = format_['Id'].apply(
        lambda x: x.split('_')[1][1:]).astype('int')
    format_.drop(columns=["Id","Prediction"], axis=1, inplace=True)

    # Get the desired data for prediction
    X_p = format_[['userId', 'movieId']].values
    X_pred = [X_p[:, 0], X_p[:, 1]]

    # Predict the ratings using the trained NN
    predictions = model.predict(X_pred)
    # round predictions
    preds_nn = [x[0] for x in predictions]

    # Transform the data into the desired format
    format_['NN_Predictions'] = preds_nn
    format_['Id'] = 'r' + format_['userId'].astype('str') + '_c' + format_[
        'movieId'].astype('str')
    format_.drop(columns=["userId", "movieId"], inplace=True)
    format_ = format_.set_index("Id")
    # Safety check
    format_ = format_.replace(6, 5)
    return format_
