# -*- coding: utf-8 -*-
import os

import pandas as pd
import numpy as np

try:
    from .processing import *
    from .NNmodel import *
except (ModuleNotFoundError, ImportError):
    from processing import *
    from NNmodel import *

def main(input_path, format_path):
    """Predicts the Netflix ratings using a Neural Network model
    
    More precisely, it preprocesses the data to make it compatible with the
    NN model then finally postprocesses the result to give predictions the desired format.

    Args:
        input_path: Path to the samples
        format_path: Path to the submission format file
        
    Returns:
        np.array: The predictions of the ratings
    """
    #preprocessing
    X_train_array, y_train, X_test_array, y_test, n_users, n_movies = preprocessing(input_path)

    #Build the NN model and train it
    n_factors = 4
    model = RecommenderNet(n_users + 1, n_movies + 1, n_factors)
    history = model.fit(x=X_train_array, y=y_train, batch_size=128, epochs=10,
                        verbose=1, validation_data=(X_test_array, y_test))

    #generating the predictions
    print("Generating predictions")
    predictions = predict(model, format_path)

    return predictions

if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    main(CURRENT_DIR + "/../data/data_train.csv",
         CURRENT_DIR + "/../data/sampleSubmission.csv")