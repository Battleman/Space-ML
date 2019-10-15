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

def preprocessing(y, x):
    """Take current features and results, clean
    """
    #splitting the samples in function of the value of PRI_jet_num 
    x_jet0=x[x[:,jet_ind]==0]
    x_jet1=x[x[:,jet_ind]==1]
    x_jet23=x[x[:,jet_ind]>1]
    
    #splitting the y values
    y_jet0=y[x[:,jet_ind]==0]
    y_jet1=y[x[:,jet_ind]==1]
    y_jet23=y[x[:,jet_ind]>1]
    
    #removing all the non-relevant columns
    x_jet0=np.delete(x_jet0,phi,1)
    x_jet1=np.delete(x_jet1,phi,1)
    x_jet23=np.delete(x_jet23,phi,1)
    
    #removing all -999 columns
    x_jet0=x_jet0[:,np.var(x_jet0,0)!=0]
    x_jet1=x_jet1[:,np.var(x_jet1,0)!=0]
    x_jet23=x_jet23[:,np.var(x_jet23,0)!=0]
    
    #replacing the outliers by the most common value of their column
    x_jet0=outlier_cleanup(x_jet0)
    x_jet1=outlier_cleanup(x_jet1)
    x_jet23=outlier_cleanup(x_jet23)
    
    return np.array([[y_jet0,x_jet0],[y_jet1,x_jet1],[y_jet23,x_jet23]])