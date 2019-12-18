# -*- coding: utf-8 -*-
import pandas as pd
import re


def preprocessing(data):
    """Preprocesses the data.

    More precisely, turns a table of format 'ri_cj|rating' into a table of format 'Row vs Col: Ratings'.

    Args:
        data: The samples

    Returns:
        np.array: The 'Row vs Col' rating matrix
    """
    # extracting row and column numbers
    data['Id'] = data['Id'].apply(lambda x: re.findall(r'\d+', str(x)))
    # turn 'Row' and 'Col' values into features
    data[['Row', 'Col']] = pd.DataFrame(
        data.Id.values.tolist(), index=data.index)
    # dropping useless features
    data = data.drop(columns='Id')
    # pivotting the table to get the desired matrix
    data = data.pivot(index='Row', columns='Col', values='Prediction')
    return data


def postprocessing(classified, format_):
    """Postprocesses the data.

    More precisely, turns the given table of format 'Row vs Col: Ratings' into a table of format 'ri_cj|rating',
    filtering out unwanted entries.

    Args:
        classified: The 'Row vs Col: Ratings' matrix
        format_: Model output format (used to rename the columns and filter-out unwanted entries)

    Returns:
        np.array: The 'ri_cj|rating' table
    """
    # converting the columns back to the 'Col' column and making 'Row' a column instead of an index
    classified = pd.melt(classified.reset_index(), id_vars=[
                         'Row'], var_name='Col', value_name='Rating')
    # converting 'Row' and 'Col' values into an id
    classified.index = 'r'+classified['Row']+'_c'+classified['Col']
    classified = classified.drop(columns=['Row', 'Col'])
    # using format_ to identify the subsample of predictions we desire
    classified = classified[classified.index.isin(list(format_['Id']))].reset_index(
    ).rename(columns={'index': 'Id', 'Rating': 'KMeans'})
    classified.set_index("Id", inplace=True)
    return classified
