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
    """Builds a Neural Network model using the Keras library. 

    Args:
        n_users: number of unique users in the dataset
        n_movies: number of unique movies in the dataset
        n_factors: number of embeddings (size of the vectors) representing users/movies in the NN model

    """
    #Build embeddings for users
    user = Input(shape=(1,))
    u_embeddings = Embedding(n_users, n_factors)(user)
    u_embeddings = Flatten()(u_embeddings)
    #Dropout to reduce overfitting
    u_embeddings = Dropout(0.02)(u_embeddings)
    
    #Build embeddings for movies
    movie = Input(shape=(1,))
    m_embeddings = Embedding(n_movies, n_factors)(movie)
    m_embeddings = Flatten()(m_embeddings)
    #Dropout to reduce overfitting
    m_embeddings = Dropout(0.02)(m_embeddings)

    merge = keras.layers.concatenate([m_embeddings, u_embeddings], axis=1)
    #Dropout to reduce overfitting
    merge = Dropout(0.03)(merge)
    
    #Fully connected layer of 100 neurons
    dense_layer1 = Dense(100, kernel_initializer='he_normal')(merge)
    dense_layer1 = Activation('relu')(dense_layer1)

    #Fully connected layer of 50 neurons
    dense_layer2 = Dense(50, kernel_initializer='he_normal')(dense_layer1)
    dense_layer2 = Activation('relu')(dense_layer2)
    
    #Fully connected layer of 10 neurons
    dense_layer3 = Dense(10, kernel_initializer='he_normal')(dense_layer2)
    dense_layer3 = Activation('relu')(dense_layer3)

    out = Dense(1, activation='relu', name='Activation')(dense_layer3)
    model = Model(inputs=[user, movie], outputs=out)
    opt = Adam(lr=0.001)
    model.compile(loss=root_mean_squared_error, optimizer=opt)
    return model