import numpy as np
import scipy.sparse as sp
import pickle as pkl


def compute_error(data, user_features, item_features, nz):
    """compute the loss (MSE) of the prediction of nonzero elements."""
    mse = 0
    for row, col in nz:
        item_info = item_features[:, row]
        user_info = user_features[:, col]
        mse += (data[row, col] - user_info.T.dot(item_info)) ** 2
    return np.sqrt(mse / len(nz))


def load_data(path_dataset):
    """Load data in text format, one rating per line."""
    with open(path_dataset, "r") as f:
        data = f.read().splitlines()[1:]
    return _preprocess_data(data)


def _preprocess_data(data):
    """Preprocessing the text data, conversion to numerical array format."""
    def deal_line(line):
        pos, rating = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row), int(col), float(rating)

    def statistics(data):
        row = set([line[0] for line in data])
        col = set([line[1] for line in data])
        return min(row), max(row), min(col), max(col)

    # parse each line
    data = [deal_line(line) for line in data]

    # do statistics on the dataset.
    min_row, max_row, min_col, max_col = statistics(data)

    # build rating matrix.
    ratings = sp.lil_matrix((max_row, max_col))
    for row, col, rating in data:
        ratings[row - 1, col - 1] = rating
    return ratings


def split_data(ratings, p_test=0.1):
    """split the ratings to training data and test data.
    Args:
        min_num_ratings:
            all users and items we keep must have at least min_num_ratings per user and per item.
    """
    # set seed
    np.random.seed(988)

    # init
    num_rows, num_cols = ratings.shape
    train = sp.lil_matrix((num_rows, num_cols))
    test = sp.lil_matrix((num_rows, num_cols))

    print("the shape of original ratings. (# of row, # of col): {}".format(
        ratings.shape))

    nz_items, nz_users = ratings.nonzero()

    # split the data
    set_nz_users = set(nz_users)
    for i, user in enumerate(set_nz_users):
        print("Splitting progression: {}%".format(
            100*(i+1)/len(set_nz_users)), end="\r")
        # randomly select a subset of ratings
        row, col = ratings[:, user].nonzero()
        selects = np.random.choice(row, size=int(len(row) * p_test))
        residual = list(set(row) - set(selects))

        # add to train set
        for r in residual:
            train[r, user] = ratings[r, user]

        # add to test set
        for s in selects:
            test[s, user] = ratings[s, user]
    print("")
    print("Total number of nonzero elements in original data:{v:,}".format(
        v=ratings.nnz))
    print("Total number of nonzero elements in train data:{v:,}".format(
        v=train.nnz))
    print(
        "Total number of nonzero elements in test data:{v:,}".format(v=test.nnz))
    return train, test
