import torch
import pandas as pd
import numpy as np
import pickle, logging, spacy, sys, os, json, requests
import matplotlib.pyplot as plt

from helpers.classes import Collection
from tqdm import tqdm
from bs4 import BeautifulSoup
from datetime import datetime

import spacy, nltk
from sklearn.feature_extraction.text import TfidfVectorizer

from helpers.cloze_generation import generate_clozes_from_point, named_entity_answer_generator as ne_answer_generator, noun_phrase_answer_generator as np_answer_generator

from helpers.table_processing import preprocess_table, read_process_table, find_relevant_column_header, find_relevant_content
from helpers.t5_language_model import summarise_t5_results




if __name__ == '__main__':
    vectorizer = TfidfVectorizer(ngram_range=(1,1))
    # vectorizer = TfidfVectorizer(ngram_range=(2,3))
    nlp_model = spacy.load('en_core_web_trf')
    stopwords = set(nltk.corpus.stopwords.words('english'))

    df = pd.read_pickle('pickles/dataset_20210625_184837.pkl')
    clozes_df = pd.read_json('pickles/clozes_20210807_165700.json')

    _cloze = clozes_df.iloc[20].source_text
    _table = clozes_df[clozes_df['source_text'] == _cloze]
    _relevant_dfs = _table.data.values[0]

    df_iterator = read_process_table(_relevant_dfs[1])

    test = next(df_iterator)
    test = next(df_iterator)

    # subdf, rows, cols, relevant_rows, relevant_cols, pairwise_matrices = find_relevant_content(_cloze, test, vectorizer)

    _cloze2 = 'West Midlands, Yorkshire and The Humber, Scotland and East Midlands, were the four regions decreasing year-on-year in approximate gross value added aGVA; the largest percentage decrease was in West Midlands where aGVA fell by £2.5 billion (2.6%), from £94.5 billion to £92 billion.'

    subdf2, rows2, cols2, relevant_rows2, relevant_cols2, pairwise_matrices2 = find_relevant_content(
        cloze=_cloze2, df=test, tfidf_vectorizer=vectorizer, spacy_nlp=nlp_model, stopwords=stopwords, return_empty=False)
    # pairwise_matrices, relevant_content, relevant_columns, relevant_rows, row_booleans = find_relevant_content(_cloze, test, vectorizer)

    print('test')