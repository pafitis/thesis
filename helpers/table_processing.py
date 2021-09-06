import numpy as np
import pandas as pd
import os

from helpers.configs import ROW_NAN_THRESHOLD, IMPUTE_COL_NANS, IMPUTE_COL_THRESHOLD, DROP_UNNAMED, FILL_NA, BACKFILL_HEADERS, IMPUTE_FIRST_COL_EMPTY, IMPUTE_ALL_COL_EMPTY, LOST_COL_LEN_THRESHOLD, SPACY_MODEL

import spacy, nltk
from sklearn.feature_extraction.text import TfidfVectorizer

from helpers.cloze_generation import generate_clozes_from_point, named_entity_answer_generator as ne_answer_generator, noun_phrase_answer_generator as np_answer_generator

# from nltk.corpus import stopwords


def lemmatize(text, spacy_nlp, stopwords = []):
    """Perform lemmatization and stopword removal in the clean text
       Returns a list of lemmas
    """
    doc = spacy_nlp(text)
    lemma_list = [str(tok.lemma_).lower() \
        if tok.is_alpha and tok.text.lower() not in stopwords else tok.text for tok in doc
                  ]
    return ' '.join(lemma_list)

def linearize_table(table, preprocess= True, include_all = False):
    '''
    outputs a pandas-read excel file (.xls/.xlsx)
    in linear format for use in LM models

    please read: https://aclanthology.org/2020.acl-main.398/
    '''
    if preprocess:
        table = preprocess_table(table)
    out_string = []
    for _row in table.iterrows():
        row_num = _row[0]
        content = _row[1]
        keys, vals = content.keys(), content.values
        if include_all:
            row_content = "; ".join([f"{k} is {v}" for k, v in zip(keys, vals)])
        else:
            row_content = "; ".join(
                [f"{k} is {v}" for k, v in zip(keys, vals) \
                    if str(v) not in ['', ' ', 'nan', 'NaN'] and k not in ['-']])
        
        out_string.append(f"Row {row_num + 1}: {row_content}.")

    return ' '.join(out_string)

def preprocess_table(table, verbose = False):
    '''
    preprocess table

    this function takes aggressive assumptions; as we do not have a consistent way to know where the data starts within each sheet. for this reason this might break at somepoint. use with caution.

    preprocessing performed
        1. remove columns that are purely NaN
        2. remove rows that are purely NaN
        3. remove rows/cols that are mostly NAN. (> than 1.5 * ROW/COLUMN_NAN_THRESHOLD from helpers.configs)
    
    
    returns processed table

    notes: expects input table to have reset index
    '''

    if not isinstance(table, pd.DataFrame):
        table = pd.DataFrame(table)
    if table.shape == (0,0):
        return
    # drop nan columns
    table = table.dropna(how = 'all', axis = 1)
    # store a version of "original" table. not original since we drop columns, but these are all empty so should be okay. reason I do this here and not before dropping is that there are issues with matching lost columns later on. We need to copy before rows are dropped due to indexing issues too, some were doubly appended.
    original_table = table
    # drop nan rows
    table = table.dropna(how = 'all', axis = 0)

    if table.shape[0] < 3 or table.shape[1] < 3:
        return
    
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
    # store what we lost in a meta_data list
    # this looks at the data we ignore and try to infer the column headers if possible; this is mostly useful in multilevel headers
    # we ignore potential headers that are more than LOST_COL_LEN_THRESHOLD which defaults at 100 chars
    _first_index = table.index[0]
    lost_content = original_table.iloc[:_first_index]
    lost_content = lost_content.dropna(how = 'all', axis = 0) # drop empty rows

    # this tries to remove rows that are mostly empty but only have 1-2 values that are not. these types of rows are usually an issue. they are often the first cell or first two cells that have comments and need to be removed. finds the rows that have less nans than the median value
    nums_of_nans_in_lost = lost_content.apply(lambda x: x.isna().sum(), axis = 1).values
    which_are_less = (nums_of_nans_in_lost < np.median(nums_of_nans_in_lost)) | \
        (nums_of_nans_in_lost < (lost_content.shape[1] - 1))
    lost_content = lost_content[which_are_less]
    lost_content = lost_content.apply(lambda x: x.ffill(), axis = 1) # forward fill for multi head
    lost_content = lost_content.astype(str).replace('nan', '') # convert nans to empty string
    # lost_content = lost_content.dropna(
    #     how = 'all', axis = 0).astype(str).replace('nan', '')
    lost_col_content = [x.strip() for x in lost_content.agg(' '.join, axis = 0).values]
    lost_col_content = [x if len(x) < LOST_COL_LEN_THRESHOLD else '' \
        for x in lost_col_content ]
    # replace empty strings to nan so we can use ffill then replace back ugly but works
    lost_col_content = pd.Series(
        lost_col_content).replace('', np.nan).ffill().replace(np.nan, '').astype('string')

    # this forward-imputes column values if they are > 0.9 * num_rows
    # this hopefully handles cases where values represent groups
    # and we want to repeatedly impute it for our LM
    # see: 'datasets/businessindustryandtrade_itandinternetindustry_datasets_ictactivityofukbusinessesecommerceandictactivity.xls'
    if IMPUTE_COL_NANS:
        if verbose:
            print(f'IMPUTE_COL_NANS: {IMPUTE_COL_NANS}')
        _nan_col_count = table.isnull().sum(axis = 0)
        _impute_where = (_nan_col_count > table.shape[0] * IMPUTE_COL_THRESHOLD)
        _cols = table.columns[_impute_where]

        table[_cols] = table[_cols].ffill()
    

    if DROP_UNNAMED:
        if verbose:
            print(f'DROP_UNNAMED: {DROP_UNNAMED}')
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
        
        # we remove the headers that are Unnamed XXX and look for the next available row to use as header name
        if BACKFILL_HEADERS:
            if verbose:
                print(f'BACKFILL_HEADERS: {BACKFILL_HEADERS}')
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
        if verbose:
            print(f'FILL_NA: {FILL_NA}')
        table = table.fillna('')
        table = table.replace('-', '')
        table.columns = table.columns.fillna('-')

    if IMPUTE_FIRST_COL_EMPTY:
        if verbose:
            print(f'IMPUTE_FIRST_COL_EMPTY: {IMPUTE_FIRST_COL_EMPTY}')
        # find where empty entries appear, change to nan so we can forward fill with appropriate values, then save inplace
        _impute_where = (table.iloc[:, 0] == '').values
        _impute_where[0] = False # change first to false as this raises issues there is nothing to forward fill with
        table.iloc[_impute_where, 0] = np.nan
        table.iloc[:, 0].ffill(inplace=True)
    if IMPUTE_ALL_COL_EMPTY:
        if verbose:
            print(f'IMPUTE_ALL_COL_EMPTY: {IMPUTE_ALL_COL_EMPTY}')
        # _impute_where = (table.iloc[:, :] == '').values
        # table.iloc[_impute_where] = np.nan
        table.replace('', np.nan, inplace= True)
        table.iloc[:, :].ffill(inplace=True)

    
    table = table.reset_index(drop = True)
    # set new headers as the concatenation of lost content and header names, also strip whitespace
    table.columns = [x.strip() \
        for x in lost_col_content + ' ' + table.columns.values.astype(str)]
    return table.astype(str)

def read_process_table(
    path, data_root = 'datasets/', file_ext = '.xls', return_iterator = True):
    # paths are saved as mainsection/subsection/../ but were saved as mainsection_subsection_...; hence we change the '/' to '_'
    if '/' in path:
        path = path.replace('/', '_')
        if path[0] == '_':
            path = path[1:]
    
    fullpath = data_root + path + file_ext
    excel_file = pd.ExcelFile(fullpath)
    sheet_names = excel_file.sheet_names

    
    if return_iterator:
        return (preprocess_table(excel_file.parse(x)) for x in sheet_names)
    else:
        return [preprocess_table(excel_file.parse(x)) for x in sheet_names]


def find_relevant_column_header(cloze, df, tfidf_vectorizer, top_k = 5):
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Relevant column extraction failed: please provide pd.DataFrame')
    
    if not isinstance(tfidf_vectorizer, TfidfVectorizer):
        raise TypeError('Please provide sklearn TF-IDF Vectorizer')
    
    
    columns = df.columns.to_list()
    _list = [cloze] + columns
    X = tfidf_vectorizer.fit_transform(_list)

    pairwise_matrix = (X * X.T).toarray()
    similarities = pairwise_matrix[0][1:]

    # reverse array so we return top_k most relevant
    most_relevant = (-similarities).argsort()[:top_k]

    non_zero_similarities = most_relevant[similarities[most_relevant] != 0.0]
    if len(non_zero_similarities):
        # returns cols idx and names
        return non_zero_similarities, columns[non_zero_similarities]
        # return [(x, columns[x]) for x in non_zero_similarities]
    else:
        return None

def find_relevant_content(cloze, df, tfidf_vectorizer, 
    spacy_nlp = None, stopwords = [], return_empty = False):

    '''
    spacy_nlp: pass spacy.load(model); if none lemmatization is not carried through
    '''
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Relevant column extraction failed: please provide pd.DataFrame')
    
    if not isinstance(tfidf_vectorizer, TfidfVectorizer):
        raise TypeError('Please provide sklearn TF-IDF Vectorizer')

    columns = df.columns.to_list()
    # unique contents of each column
    # len(contents) = num_of_columns
    # each element of contents = num_of_unique_column_elements
    # we add cloze + header + column_contents 
    contents = [
        [cloze] + [col] + df.astype(str).iloc[:, i].unique().tolist() \
        for i, col in enumerate(columns)] # use iloc instead of loc as sometimes colnames are not unique
    
    # construct list of spacy docs for each column
    if spacy_nlp is not None:
        contents = [
            [lemmatize(x,spacy_nlp, stopwords) for x in col_contents] \
                for col_contents in contents
            ]


    Xs = [tfidf_vectorizer.fit_transform(x) for x in contents]
    pairwise_matrices = [(x * x.T).toarray()[0] for x in Xs] #slice [0] to get relations with cloze

    
    relevant_content = []
    relevant_rows = set()
    
    for col_idx, column in enumerate(columns):
        # find content that has non-zero tfidf with the cloze (also ignore 1.0 tfidf because thats with itself)
        _relevant_content_ix = np.where(
            (pairwise_matrices[col_idx][1:] != 0.0))[0]
            # (pairwise_matrices[col_idx][1:] != 0.0) * \
            #      (pairwise_matrices[col_idx][1:] != 1.0))

        if len(_relevant_content_ix):
            # find the column values that are related with the cloze
            column_content = contents[col_idx][1:] # slice 1: to remove appended cloze
            _relevant_content = [column_content[x] for x in _relevant_content_ix]
            # store
            relevant_content.append(_relevant_content)
            # find the rows of the dataset that contain these values
            _relevant_rows = [df.iloc[:, col_idx] == subitem for subitem in _relevant_content]

            row_bools = np.where(np.sum(_relevant_rows, axis = 0))[0]
            relevant_rows.update(row_bools)
            # if len(_relevant_rows) == 3:
            #     # we slice [1] because we pass a 2d list, and [1] represents the rows where value == True;
            #     row_bools = np.where(_relevant_rows)[1]
            #     relevant_rows.update(row_bools)

    relevant_columns = [True if len(x) else False for x in relevant_content]
    relevant_columns = np.where(relevant_columns)[0]
    relevant_rows = np.array(list(relevant_rows))

    cols = relevant_columns if len(relevant_columns) != 0 \
        else [x for x in range(df.shape[1])]
    rows = relevant_rows if len(relevant_rows) != 0 else df.index.tolist()
    subdf = df.iloc[rows, cols].reset_index(drop = True)

    if return_empty:
       pass 
    return subdf, rows, cols, relevant_rows, relevant_columns, pairwise_matrices, contents
    
    # return df.iloc[relevant_rows, relevant_columns], relevant_rows, relevant_columns



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

    # filepath = 'datasets/businessindustryandtrade_itandinternetindustry_datasets_ictactivityofukbusinessesecommerceandictactivity.xls'

    # filepath = 'datasets/economy_environmentalaccounts_datasets_seminaturalhabitatsecosystemservices.xls'

    # filepath = 'datasets/businessindustryandtrade_itandinternetindustry_datasets_ictactivityofukbusinessesecommerceandictactivity.xls'

    # good example to see BACKFILL_HEADERS
    # sheet_name 3
    # filepath = 'datasets/employmentandlabourmarket_peopleinwork_labourproductivity_datasets_subregionalproductivitylabourproductivitygvaperhourworkedandgvaperfilledjobindicesbylocalenterprisepartnership.xls'
    # but this fails:
    # 'datasets/peoplepopulationandcommunity_wellbeing_datasets_measuringnationalwellbeingdomainsandmeasures.xls'
    

    # filepath = 'datasets/peoplepopulationandcommunity_wellbeing_datasets_measuringnationalwellbeingdomainsandmeasures.xls'

    # filepath = 'datasets/businessindustryandtrade_business_businessservices_datasets_uknonfinancialbusinesseconomyannualbusinesssurveyrevisionsandchangeonpreviousyear.xls'
    
    # table_data = pd.ExcelFile(filepath)
    # sheet_names = table_data.sheet_names

    # df = table_data.parse(sheet_names[2])
    # # df = table_data.parse(sheet_names[1])
    # processed = preprocess_table(df)

    # path = '/businessindustryandtrade/changestobusiness/mergersandacquisitions/datasets/mergersandacquisitionsinvolvingukcompanies'

    # path = '/peoplepopulationandcommunity/birthsdeathsandmarriages/deaths/datasets/standardofproofsuicidedata'

    path = 'businessindustryandtrade_business_businessservices_datasets_uknonfinancialbusinesseconomyannualbusinesssurveyrevisionsandchangeonpreviousyear'

    test = pd.ExcelFile('datasets/' + path+ '.xls')
    preprocess_table(test.parse('ABS Revisions to Data'))

    # test linearizer
    # data = {'Actors': ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
    #     'Age': ["56", "45", "59"],
    #     'Number of movies': ["87", "53", "69"]
    # }
    # table = pd.DataFrame.from_dict(data)
    # linear_table = linearize_table(processed)