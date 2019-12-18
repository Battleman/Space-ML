import numpy as np
import scipy.sparse as sp
import pickle as pkl
from surprise import Dataset
from surprise import Reader
import re
import pandas as pd

def csv_to_data(file_path):
    """Read a CSV file and change it to data usable by surprize.

    Arguments:
        file_path {str} -- path of the file to read

    Returns:
        trainset -- data in usable format
    """
    reader = Reader(line_format='user item rating', sep=",")
    data = Dataset.load_from_file(file_path, reader=reader)
    trainset = data.build_full_trainset()
    return trainset


def pandas_to_data(df):
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
    data.loc[:, 'Id'] = data.loc[:, 'Id'].apply(
        lambda x: re.findall(r'\d+', str(x)))
    # turn 'Row' and 'Col' values into features
    data[['Row', 'Col']] = pd.DataFrame(
        data.Id.values.tolist(), index=data.index)
    # dropping useless features
    data = data.drop(columns='Id')
    return data
