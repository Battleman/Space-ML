import numpy as np

def augment(x, total_degree):
    
    assert total_degree > 1
    
    # add a column of ones
    x_aug = np.append(np.ones((len(x),1)), x, axis=1)
    
    #double X in the third dimension
    x3d = np.array([x]*2)
    
    # augment for each degree
    num_feats = x.shape[1]
    for deg in range(2,total_degree+1):
        print("Degree {}".format(deg))
        raw_combinations = np.stack(np.meshgrid(*[range(num_feats)]*deg), -1).reshape((-1,deg))
        unique_combinations = []
        for comb in raw_combinations: # comb = [0, 2, 1] e.g.
            # only keep ordered sequences, as they are all unique
            if all(comb[i] <= comb[i+1] for i in range(len(comb)-1)):
                unique_combinations.append(comb)
        # magic
        x_aug = np.append(x_aug, np.prod(x3d[0,:,unique_combinations], axis=1).T, axis=1)
    return x_aug