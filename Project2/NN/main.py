# -*- coding: utf-8 -*-
import os
import pandas as pd
try:
    from .processing import preprocessing, predict
    from .NNmodel import NN_model
except (ModuleNotFoundError, ImportError):
    from processing import preprocessing, predict
    from NNmodel import NN_model


def main(input_, format_):
    """Predicts the Netflix ratings using a Neural Network model

    More precisely, it preprocesses the data to make it compatible with the
    NN model which is then trained and finally postprocesses the result
    to give predictions the desired format.

    Args:
        input_: The samples
        format_: The submission format file

    Returns:
        np.array: The predictions of the ratings
    """
    # preprocessing
    X_all, y, n_users, n_movies = preprocessing(input_)

    # Build the NN model and train it
    n_factors = 4
    model = NN_model(n_users + 1, n_movies + 1, n_factors)
    _ = model.fit(x=X_all, y=y, batch_size=128, epochs=10,
                  verbose=1)

    # generating the predictions
    print("Generating predictions ...")
    predictions = predict(model, format_)

    return predictions


if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    main(pd.read_csv(CURRENT_DIR + "/../data/data_train.csv"),
         pd.read_csv(CURRENT_DIR + "/../data/sampleSubmission.csv"))
