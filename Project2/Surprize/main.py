import numpy as np
import surprise as spr
from surprise import Dataset
from surprise import Reader
import os
import pandas as pd
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import time
import re
import sys

try:
    from .helpers import compute_error, split_data, load_data
except ImportError:
    from helpers import compute_error, split_data, load_data


def get_all_ids(path):
    """Read the provided path and return a list of all ids.

    Arguments:
        path {str} -- Path to the file to read

    Returns:
        list -- list of ids
    """
    raw_ids = []
    with open(path) as f1:
        for l in f1.readlines()[1:]:
            id, _ = l.split(",")
            raw_ids.append(id)
    return raw_ids


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


def predictor(ids_chunk):
    """Elementary function to parallelize predictions.

    Arguments:
        ids_chunk {list} -- List of ids that must be predicted

    Returns:
        list -- list of (id, rating)
    """
    print("Working on a chunk")
    res_chunk = []
    for i in ids_chunk:
        uid, iid = get_ids(i)
        p = algo_in_use.predict(uid, iid)
        res_chunk.append((i, p.est))
    print("Finished chunk")
    return res_chunk


def parallelize_predictions(ids, n_cores=16):
    """Split work through cores to create predictions efficiently.

    Arguments:
        ids {list} -- All ids to predict

    Keyword Arguments:
        n_cores {int} -- In how many chunks split the ids (default: {16})

    Returns:
        list -- list of (id, rating) for all ids
    """
    splitted_ids = np.array_split(ids, n_cores)
    pool = Pool(n_cores)
    res = np.concatenate(pool.map(predictor, splitted_ids))
    res = [(r[0], float(r[1])) for r in res]
    pool.close()
    pool.join()
    return res


def create_3cols(orig_filename, name):
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    filename_3cols = CURRENT_DIR+"/cache/{}_3cols.csv".format(name)
    with open(orig_filename) as f1:
        with open(filename_3cols, "wt") as f2:
            for l in f1.readlines()[1:]:
                id, rating = l.split(",")
                row, col = id.split("_")
                row = row[1:]
                col = col[1:]
                f2.write("{},{},{}".format(row, col, rating))
    return filename_3cols


def preprocess_df(data):
    data.loc[:, 'Id'] = data.loc[:, 'Id'].apply(
        lambda x: re.findall(r'\d+', str(x)))
    # turn 'Row' and 'Col' values into features
    data[['Row', 'Col']] = pd.DataFrame(
        data.Id.values.tolist(), index=data.index)
    # dropping useless features
    data = data.drop(columns='Id')
    return data


def main(train_df, target_df, cache_name="test"):
    global algo_in_use
    CACHED_DF_FILENAME = os.path.dirname(
        os.path.abspath(__file__)) +\
        "/cache/cached_predictions_{}.pkl".format(cache_name)
    train_df = preprocess_df(train_df)
    trainset = pandas_to_data(train_df)
    ids_to_predict = target_df["Id"].to_list()

    # try to retrieve backup dataframe
    try:
        print("Retrieving cached predictions")
        all_algos_preds_df = pd.read_pickle(CACHED_DF_FILENAME)
        print("Ensuring cached IDs match given IDs")
        assert sorted(ids_to_predict) == sorted(all_algos_preds_df.index.values)
        print("Indices match, continuing")
    except (FileNotFoundError, AssertionError):
        print("No valid cached predictions found")
        all_algos_preds_df = pd.DataFrame(ids_to_predict, columns=["Id"])
        all_algos_preds_df.set_index("Id", inplace=True)

    all_algos = {"SVD": spr.SVD(),
                 "Baseline": spr.BaselineOnly(),
                 "NMF": spr.NMF(),
                 "Slope One": spr.SlopeOne(),
                 "KNN Basic": spr.KNNBasic(),
                 "KNN Means": spr.KNNWithMeans(),
                 "KNN Baseline": spr.KNNBaseline(),
                 "KNN Zscore": spr.KNNWithZScore(),
                 "SVD ++": spr.SVDpp()}

    for name in all_algos:
        print("##### {} ####".format(name))
        if name in all_algos_preds_df.columns:
            print("Already computed {}, skipping".format(name))
            continue
        algo = all_algos[name]
        time.sleep(1)
        algo.fit(trainset)
        time.sleep(1)
        algo_in_use = algo
        print("Generating predictions...")
        predictions = parallelize_predictions(ids_to_predict, 80)
        print("Done. Merging with previous results")
        this_algo_preds_df = pd.DataFrame(predictions, columns=["Id", name])
        this_algo_preds_df.set_index("Id", inplace=True)
        all_algos_preds_df = pd.merge(all_algos_preds_df, this_algo_preds_df,
                              left_index=True, right_index=True)
        all_algos_preds_df.to_pickle(CACHED_DF_FILENAME)
    print("DONE computing surprize")
    return all_algos_preds_df
