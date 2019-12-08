# -*- coding: utf-8 -*-
import pandas as pd
import re

def preprocessing(path):
    """Preprocesses the data.

    More precisely, loads a table of format 'ri_cj|rating' at the given path 
    and turns it into a table of format 'Row vs Col: Ratings'.

    Args:
        path: Path to the data
        
    Returns:
        np.array: The 'Row vs Col' rating matrix
    """
    #loading the training data
    data=pd.read_csv(path)
    #extracting row and column numbers
    data['Id']=data['Id'].apply(lambda x: re.findall(r'\d+', str(x)))
    #turn 'Row' and 'Col' values into features
    data[['Row', 'Col']]=pd.DataFrame(data.Id.values.tolist(), index= data.index)
    #dropping useless features
    data=data.drop(columns='Id')
    #pivotting the table to get the desired matrix
    data=data.pivot(index='Row', columns='Col', values='Prediction')
    return data

def postprocessing(classified, format_path):
    """Postprocesses the data.

    More precisely, turns the given table of format 'Row vs Col: Ratings' into a table of format 'ri_cj|rating',
    filtering out unwanted entries.

    Args:
        classified: The 'Row vs Col: Ratings' matrix
        format_path: Path to the model output (used to rename the columns and filter-out unwanted entries)
        
    Returns:
        np.array: The 'ri_cj|rating' table
    """
    #converting the columns back to the 'Col' column and making 'Row' a column instead of an index
    classified=pd.melt(classified.reset_index(), id_vars=['Row'], var_name='Col', value_name='Rating')
    #converting 'Row' and 'Col' values into an id
    classified.index='r'+classified['Row']+'_c'+classified['Col']
    classified=classified.drop(columns=['Row','Col'])
    #loading the sample submission data to identify the subsample of the prediction we desire
    sample_sumbission=pd.read_csv(format_path)
    classified=classified[classified.index.isin(list(sample_sumbission['Id']))].reset_index().rename(columns={'index': 'Id', 'Rating':'Prediction'})
    return classified