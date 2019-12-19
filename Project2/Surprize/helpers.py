"""Helpers for surprise framework"""
from surprise import Dataset
from surprise import Reader
import re
import pandas as pd


def pandas_to_data(df):
    """Input and pandas Dataframe and output a usable trainset."""
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["Row", "Col", "Prediction"]],
                                reader=reader)
    trainset = data.build_full_trainset()
    return trainset


def get_ids(rid):
    """From a raw id (rX_cY) yield user and item ids (X and Y)

    Arguments:
        rid {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    u, i = rid.split("_")
    return u[1:], i[1:]


def preprocess_df(data):
    """Preprocess a dataframe to split user and item IDs."""
    data.loc[:, 'Id'] = data.loc[:, 'Id'].apply(
        lambda x: re.findall(r'\d+', str(x)))
    # turn 'Row' and 'Col' values into features
    data[['Row', 'Col']] = pd.DataFrame(
        data.Id.values.tolist(), index=data.index)
    # dropping useless features
    data = data.drop(columns='Id')
    return data
