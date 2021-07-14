import numpy as np
import pandas as pd
import os

from helpers.configs import ROW_NAN_THRESHOLD, IMPUTE_COL_NANS, IMPUTE_COL_THRESHOLD, DROP_UNNAMED, FILL_NA, BACKFILL_HEADERS

def linearize_table(table, preprocess= True):
    '''
    outputs a pandas-read excel file (.xls/.xlsx)
    in linear format for use in LM models

    please read: https://aclanthology.org/2020.acl-main.398/
    '''
    if preprocess:
        table = preprocess_table(table)
    
    print('t')
    print('t')

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
    # I am also adding +1 to the counts, to catch errors when the median
    # value is 0. If we shift by one then this multiplication will be nonzero
    # I only add this to the median calculation.
    _nan_row_count = table.isnull().sum(axis = 1)
    _nan_row_threshold = ROW_NAN_THRESHOLD * np.median(_nan_row_count + 1)
    
    # okay actually this still breaks try this;
    if _nan_row_threshold <= 2:
        _nan_row_threshold = table.shape[1] * 0.7

    table = table.drop(
        table.index[_nan_row_count > _nan_row_threshold]
        # I have thought about what happens when median == 0
        # it should be okay though as I am using a strict ineq
    )

    # this forward-imputes column values if they are > 0.9 * num_rows
    # this hopefully handles cases where values represent groups
    # and we want to repeatedly impute it for our LM
    # see: 'datasets/businessindustryandtrade_itandinternetindustry_datasets_ictactivityofukbusinessesecommerceandictactivity.xls'
    if IMPUTE_COL_NANS:
        _nan_col_count = table.isnull().sum(axis = 0)
        _impute_where = (_nan_col_count > table.shape[0] * IMPUTE_COL_THRESHOLD)
        _cols = table.columns[_impute_where]

        table[_cols] = table[_cols].ffill()

    if DROP_UNNAMED:
        # this caused some issues because the columns are returned in an Index object which causes some weird things
        # specifically if you compare float/ints to str it returns a nan but is not a nantype so careful error catching must be implemented
        # this approach of converting to str() seems to bypass this
        # I leave original solutions (which break in some scenarios) for future reference
        columns = table.columns.values
        while any(['Unnamed' in str(x) for x in columns]):
        # while any(table.columns.dropna().str.contains('Unnamed')):
        # while any(table.columns.dropna().str.contains('Unnamed').dropna()):
            table.columns = table.iloc[0]
            table = table[1:]
            columns = table.columns.values
        if BACKFILL_HEADERS:
            # find which headers are nan
            _impute_where = table.columns.isnull()
            # find values of the next available entries
            _impute_with = table.loc[:, _impute_where].iloc[0].values
            
            # replace them
            table.columns = table.columns.fillna('replace_me')
            _cols = table.columns.values
            _cols[_cols == 'replace_me'] = _impute_with
            # save
            table.columns = _cols

    
    if FILL_NA:
        table = table.fillna('-')


    return table




## NOTES

# this was giving errors. its a valid .xls file but has compatibility issues with xlrd the package pandas uses to parse excel files
    # 'datasets/businessindustryandtrade_changestobusiness_mergersandacquisitions_datasets_mergersandacquisitionsuk.xls'

# this is interesting because of how its formated
# can you think of a way to extract these sub-frames?
# maybe impute NaNs with the previous non-NaN value?
    # 'datasets/businessindustryandtrade_itandinternetindustry_datasets_ictactivityofukbusinessesecommerceandictactivity.xls'


# good example
# sheet_name 3 is working beautifully
# sheet_name 4 is already in a good format and our function doesn't break it; good-job me!
# sheet_names 1, 2 are not
# 'datasets/peoplepopulationandcommunity_healthandsocialcare_causesofdeath_datasets_alcoholspecificdeathsintheukmaindataset.xls'


if __name__ == '__main__':

    filepath = 'datasets/businessindustryandtrade_itandinternetindustry_datasets_ictactivityofukbusinessesecommerceandictactivity.xls'

    filepath = 'datasets/economy_environmentalaccounts_datasets_seminaturalhabitatsecosystemservices.xls'

    # filepath = 'datasets/businessindustryandtrade_itandinternetindustry_datasets_ictactivityofukbusinessesecommerceandictactivity.xls'

    # good example to see BACKFILL_HEADERS
    # sheet_name 3
    filepath = 'datasets/employmentandlabourmarket_peopleinwork_labourproductivity_datasets_subregionalproductivitylabourproductivitygvaperhourworkedandgvaperfilledjobindicesbylocalenterprisepartnership.xls'
    # but this fails:
    # 'datasets/peoplepopulationandcommunity_wellbeing_datasets_measuringnationalwellbeingdomainsandmeasures.xls'
    

    filepath = 'datasets/peoplepopulationandcommunity_wellbeing_datasets_measuringnationalwellbeingdomainsandmeasures.xls'
    
    table_data = pd.ExcelFile(filepath)
    sheet_names = table_data.sheet_names

    df = table_data.parse(sheet_names[3])
    processed = preprocess_table(df)

    test = linearize_table(df)
    print('test')
    print('test')
    print('test')