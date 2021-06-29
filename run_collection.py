import numpy as np
import pandas as pd
import requests
import logging
import sys
import os
import pickle
import spacy

from datetime import datetime
from bs4 import BeautifulSoup
from requests import api
from requests.models import RequestEncodingMixin
from tqdm import tqdm
from tqdm.cli import main

# from helpers.text_processing import mainpoints_entity_processing
from helpers.classes import Collection, setup_logger


if __name__ == '__main__':

    if not os.path.exists('logs/'):
        os.makedirs('logs/')
    if not os.path.exists('pickles/'):
        os.makedirs('pickles/')

    setup_logger('info')


    API_URL = 'https://api.beta.ons.gov.uk/v1/datasets/'
    collection = Collection()
    # collection.get_sections()
    collection.collect_bulletin()
    collection.process_collection()
    
    

    # RECURSION LIMIT NEEDS TO INCREASE (PROBABLY)
    # I make sure to reset it to 1e3
    sys.setrecursionlimit(10000)
    with open(
        f"pickles/collection_{str(datetime.now().strftime('%Y%m%d_%H%M%S'))}.pkl", 'wb') as f:
        pickle.dump(collection, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('collection saved')
    sys.setrecursionlimit(1000)