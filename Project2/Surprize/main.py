import numpy as np
import surprise as spr
import os
import pandas as pd
from multiprocessing import Pool
import time

try:
    from .helpers import pandas_to_data, get_ids, preprocess_df
except ImportError:
    from helpers import pandas_to_data, get_ids, preprocess_df


def predictor(ids_chunk):
    """Elementary function to parallelize predictions.

    Arguments:
        ids_chunk {list} -- List of ids that must be predicted

    Returns:
        list -- list of (id, rating)
    """
    res_chunk = []
    for i in ids_chunk:
        uid, iid = get_ids(i)
        p = algo_in_use.predict(uid, iid)
        res_chunk.append((i, p.est))
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


def main(train_df, target_df, cache_name="test", force_recompute=[]):
    """Train multiple models on train_df and predicts target_df

    Predictions are cached. If the indices don't match the indices of
    target_df, the cache is discarded.

    By default, if a method was already computed it is not recomputed again
    (except if the method name is listed in force_recompute). cache_name
    is the name to use to read and write the cache.

    Arguments:
        train_df {dataframe} -- Training dataframe
        target_df {dataframe} -- Testing dataframe

    Keyword Arguments:
        cache_name {str} -- Name to use for caching (default: {"test"})
        force_recompute {list} -- Name(s) of methods to recompute, whether or
        not it was already computed. Useful to only recompute single methods
        without discarding the rest. (default: {[]})

    Returns:
        Dataframe -- Dataframe with predictions for each methods as columns,
        IDs as indices
    """
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
        assert sorted(ids_to_predict) == sorted(
            all_algos_preds_df.index.values)
        print("Indices match, continuing")
    except (FileNotFoundError, AssertionError):
        print("No valid cached predictions found")
        all_algos_preds_df = pd.DataFrame(ids_to_predict, columns=["Id"])
        all_algos_preds_df.set_index("Id", inplace=True)

    all_algos = {"SVD": spr.SVD(n_factors=200, n_epochs=100),
                 "Baseline": spr.BaselineOnly(),
                 "NMF": spr.NMF(n_factors=30, n_epochs=100),
                 "Slope One": spr.SlopeOne(),
                 "KNN Basic": spr.KNNBasic(k=60),
                 "KNN Means": spr.KNNWithMeans(k=60),
                 "KNN Baseline": spr.KNNBaseline(),
                 "KNN Zscore": spr.KNNWithZScore(k=60),
                 "SVD ++": spr.SVDpp(n_factors=40, n_epochs=100),
                 "Co Clustering": spr.CoClustering()}

    for name in all_algos:
        print("##### {} ####".format(name))
        if name in force_recompute and name in all_algos_preds_df.columns:
            all_algos_preds_df.drop(name, axis=1, inplace=True)
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
