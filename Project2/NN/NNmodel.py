# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

import keras
from keras.models import Model
from keras import backend as K
from keras.layers import Concatenate, Dense, Dropout, Flatten
from keras.layers import Input, Reshape
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import Add, Activation, Lambda

def root_mean_squared_error(y_true, y_pred):
    """Creates a RMSE loss function for the NN training

    Args:
        y_true: list of true values of the labels
        y_pred: list of predicted labels

    """
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def NN_model(n_users, n_movies, n_factors):
    """Builds a Neural Network model using the Keras library

    Args:
        n_users: number of unique users in the dataset
        n_movies: number of unique movies in the dataset
        n_factors: number of embeddings (size of the vectors) representing users/movies in the NN model

    """
    user = Input(shape=(1,))
    u = Embedding(n_users, n_factors)(user)
    u = Flatten()(u)
    u = Dropout(0.02)(u)

    movie = Input(shape=(1,))
    m = Embedding(n_movies, n_factors)(movie)
    m = Flatten()(m)
    m = Dropout(0.02)(m)

    x = keras.layers.concatenate([m, u], axis=1)

    x = Dropout(0.03)(x)

    x = Dense(100, kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)

    x = Dense(50, kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)

    x = Dense(10, kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)

    x = Dense(1, activation='relu', name='Activation')(x)
    model = Model(inputs=[user, movie], outputs=x)
    opt = Adam(lr=0.001)
    model.compile(loss=root_mean_squared_error, optimizer=opt)
    return model