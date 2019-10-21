import numpy as np

#indicator of the missing values
undef_value=-999
#index of the splitter (PRI_jet_num)
jet_ind=22
#index of all the non-relevant columns (post-split)
phi=np.array([jet_ind,15,18,20])

#returns the most common value in x, -999 excluded
def most_common(x):
    temp=np.unique(x[x!=undef_value], return_counts=True)
    return temp[0][temp[1].argmax()]

#replaces all -999 of x by the most common value of the corresponding column (-999 excluded)
def outlier_cleanup(x):
    com_vector=np.apply_along_axis(most_common, 0, x)
    com_matrix=np.repeat(com_vector[np.newaxis,:],np.array([x.shape[0]]),axis=0)
    return np.where(x!=undef_value,x,com_matrix)

def preprocessing(x):
    """Take current features and results, clean
    """
    #splitting the samples in function of the value of PRI_jet_num 
    masks = [
        x[:,jet_ind]==0,
        x[:,jet_ind]==1,
        x[:,jet_ind]>1
    ]
    
    split_cleaned = []
    for m in masks:
        subset = x[m]
        subset = np.delete(subset, phi, 1)
        subset = subset[:,np.var(subset,0) != 0]
        subset = outlier_cleanup(subset)
        split_cleaned.append(subset)

    
    return (split_cleaned, masks)
