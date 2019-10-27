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


def augment(x, total_degree=None, simple_degree=None, tan_hyp_deg=None, ilog_deg=None, root_deg=None):

    assert total_degree > 1

    x_only_pos = x - x.min(axis=0)
    x_std = standardize(x.copy())

    # # add a column of ones
    x_aug = np.append(np.ones((len(x_std), 1)), x_std, axis=1)

    if total_degree is not None and total_degree > 1:
        # double X in the third dimension
        x3d = np.array([x_std]*2)

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

    # append simple degrees
    if simple_degree is not None and simple_degree > 1:
        print("Adding {} simple powers".format(simple_degree))
        min_deg = max(2, (total_degree if total_degree is not None else 0))
        additional_powers = np.concatenate(
            [np.power(x_std, deg) for deg in range(min_deg, simple_degree+1)], axis=1)
        x_aug = np.append(x_aug, additional_powers, axis=1)

    # append simple roots
    if root_deg is not None and root_deg > 1:
        print("Adding Roots powers")
        for deg in range(2, root_deg+1):
            x_aug = np.append(x_aug, np.power(x_only_pos, 1/deg), axis=1)

    # compute hyperbolic tan and its powers
    if tan_hyp_deg is not None and tan_hyp_deg > 0:
        print("Adding tanh powers")
        tanh = np.tanh(x)
        for deg in range(1, tan_hyp_deg+1):
            x_aug = np.append(x_aug, standardize(np.power(tanh, deg)), axis=1)
        del tanh

    # compute inverse log and append its powers
    if ilog_deg is not None and ilog_deg > 0:
        print("Adding inverse log")
        ilog = inv_log(x_only_pos)
        for deg in range(1, ilog_deg+1):
            x_aug = np.append(x_aug, np.power(ilog, deg), axis=1)
        del ilog

    return x_aug
