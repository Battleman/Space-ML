import numpy as np
import pandas as pd
try:
    from .helpers import compute_error, split_data, load_data
except ImportError:
    from helpers import compute_error, split_data, load_data


def main(path_train, path_predictions, rounded=True):
    ratings = load_data(path_train)
    sample_submit = load_data(path_predictions)

    nnz_row, nnz_col = ratings.nonzero()
    nnz_pos = list(zip(nnz_row, nnz_col))

    global_mean = np.mean([ratings[pos] for pos in nnz_pos])

    # row == user == 10k
    # col == item == 1k
    ubias = []
    ibias = []
    num_users, num_items = ratings.shape
    print(ratings.shape)
    for user_index in range(num_users):
        # find the non-zero ratings for each user in the training dataset
        user_ratings = ratings[user_index, :]
        nonzeros_user_ratings = user_ratings[user_ratings.nonzero()]

        # calculate the mean if the number of elements is not 0
        if nonzeros_user_ratings.shape[0] != 0:
            ubias.append(nonzeros_user_ratings.mean() - global_mean)

    for item_index in range(num_items):
        item_ratings = ratings[:, item_index]
    #         print(item_ratings)
        nonzeros_item_ratings = item_ratings[item_ratings.nonzero()]
    #         print(nonzeros_item_ratings)
        # calculate the mean if the number of elements is not
        bias = []
        if nonzeros_item_ratings.shape[0] != 0:
            for r in item_ratings.nonzero()[0]:
                bias.append(ratings[r, item_index] - global_mean - ubias[r])
            ibias.append(np.mean(bias))

    nnz_row_final, nnz_col_final = sample_submit.nonzero()

    predictions = [("r{}_c{}".format(r, c),
                    (np.clip(np.round(global_mean + ubias[r] + ibias[c]), 1, 5) if rounded else global_mean + ubias[r] + ibias[c]))
                   for r, c in zip(nnz_row_final, nnz_col_final)]
    return pd.DataFrame(predictions, columns=["Id", "Prediction"])
