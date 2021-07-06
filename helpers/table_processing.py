import numpy as np
import pandas as pd
import os

from helpers.configs import ROW_NAN_THRESHOLD, COLUMN_NAN_THRESHOLD

def linearize_table(table, preprocess= True):
    '''
    outputs a pandas-read excel file (.xls/.xlsx)
    in linear format for use in LM models

    please read: https://aclanthology.org/2020.acl-main.398/
    '''
    if preprocess:
        table = preprocess_table(table)
    
    pass


def preprocess_table(table):
    '''
    preprocess table

    this function takes aggressive assumptions; as we do not have a consistent way to know where the data starts within each sheet. for this reason this might break at somepoint. use with caution.

    preprocessing performed
        1. remove columns that are purely NaN
        2. remove rows that are purely NaN
        3. remove rows/cols that are mostly NAN. (> than 1.5 * ROW/COLUMN_NAN_THRESHOLD from helpers.configs)
    
    
    returns processed table
    '''

    if not isinstance(table, pd.DataFrame):
        table = pd.DataFrame(table)
    
    # drop nan columns
    table = table.dropna(how = 'all', axis = 1)
    # drop nan rows
    table = table.dropna(how = 'all', axis = 0)
    # count nans in rows and drop those rows/cols that are 1.5x the median
    # count, construct threshold, find indeces above threshold, drop
    _nan_row_count = table.isnull().sum(axis = 1)
    _nan_col_count = table.isnull().sum(axis = 0)
    _nan_row_threshold = ROW_NAN_THRESHOLD * np.median(_nan_row_count)
    _nan_col_threshold = COLUMN_NAN_THRESHOLD * np.median(_nan_col_count)
    table = table.drop(
        table.index[_nan_row_count > _nan_row_threshold]
        # I have thought about what happens when median == 0
        # it should be okay though as I am using a strict ineq
    )

    return table

# this was giving errors. its a valid .xls file but has compatibility issues with xlrd the package pandas uses to parse excel files
    # 'datasets/businessindustryandtrade_changestobusiness_mergersandacquisitions_datasets_mergersandacquisitionsuk.xls'