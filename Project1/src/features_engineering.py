import numpy as np


def inv_log(x):
    return np.log(1 / (1 + x))


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x


def augment(x, total_degree, simple_degree, tan_hyp_deg, ilog_deg):

    assert total_degree > 1

    x_only_pos = x - x.min(axis=0)

    # add a column of ones
    x_aug = np.append(np.ones((len(x), 1)), x, axis=1)

    # double X in the third dimension
    x3d = np.array([x]*2)

    # augment for each degree
    num_feats = x.shape[1]
    for deg in range(2, total_degree+1):
        print("Degree {}".format(deg))
        raw_combinations = np.stack(np.meshgrid(
            *[range(num_feats)]*deg), -1).reshape((-1, deg))
        unique_combinations = []
        for comb in raw_combinations:  # comb = [0, 2, 1] e.g.
            # only keep ordered sequences, as they are all unique
            if all(comb[i] <= comb[i+1] for i in range(len(comb)-1)):
                unique_combinations.append(comb)
        # magic
        x_aug = np.append(x_aug,
                          np.prod(x3d[0, :, unique_combinations],
                                  axis=1).T,
                          axis=1)

    # # append simple degrees
    # print("Adding simple powers")
    # for deg in range(2, simple_degree+1):
    #     print(deg)
    #     x_aug = np.append(x_aug, np.power(x, deg), axis=1)

    # # compute hyperbolic tan and its powers
    # print("Adding tanh powers")
    # tanh = np.tanh(x)
    # for deg in range(1, tan_hyp_deg+1):
    #     x_aug = np.append(x_aug, np.power(tanh, deg), axis=1)
    # del tanh

    # # compute inverse log and append its powers
    # print("Adding inverse log")
    # ilog = inv_log(x_only_pos)
    # for deg in range(1, ilog_deg+1):
    #     x_aug = np.append(x_aug, np.power(ilog, deg), axis=1)
    # del ilog

    return standardize(x_aug)
