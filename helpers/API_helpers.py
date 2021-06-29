import numpy as np
import pandas as pd
import requests
import logging
import sys
import os
import pickle
import spacy
import attr

from datetime import datetime
from bs4 import BeautifulSoup
from requests import api
from requests.models import RequestEncodingMixin
from tqdm import tqdm
from tqdm.cli import main

# from helpers.text_processing import mainpoints_entity_processing

def setup_logger(log):
    logging.basicConfig(
        filename=f"logs/{str(datetime.now().strftime('%Y%m%d_%H%M%S'))}.log",
        filemode="w",
        level=getattr(logging, log.upper()),
        datefmt="%H:%M:%S",
        format="%(asctime)s - %(levelname)s: %(message)s",
    )

class Dataset:
    def __init__(self, DATASET_ID: str, API_URL = 'https://api.beta.ons.gov.uk/v1/datasets/'):
        # remove / incase it was provided
        if DATASET_ID[0] == '/':
            print('DATASET_ID provided starts with "/". \nAutomatically removed.')
            DATASET_ID = DATASET_ID[1:]
        # api call
        call = requests.get(API_URL + DATASET_ID)
        # check if it exists
        if call.status_code != 200:
            raise LookupError(f'DATASET_ID not found. Returned status_code: {call.status_code}')

        json_info = call.json()
        self._json = json_info
        self._id = json_info.get('id')
        self._title = json_info.get('title')
        
        self.links = json_info.get('links')
        self.edition = requests.get(self.links.get('latest_version')['href']).json()
        self.description = json_info.get('description')
        self.publications = json_info.get('publications')
        self.related_datasets = json_info.get('related_datasets')

    def _initialise_publication(self):
        self.publications = self._json.get('publications')
        if self.publications is None:
            qmi_link = self._json.get('qmi').get('href')
            # hacky way, I'll try to explain so I don't forget
            # the qmi link seems to follow the naming scheme the
            # corresponding datasets have.
            # what I am doing here is leveraging the fact that
            # I can extract the dataset path by looking at the url
            # and switching "methodologies" for "datasets"
            # additionally I don't need the parameters/ anchor
            # right of /datasets/...
            dataset_link = qmi_link[:qmi_link.find('methodologies')]


class Collection:
    def __init__(self,
    API_URL = 'https://api.beta.ons.gov.uk/v1/datasets?limit=1000', ONS_URL = 'https://www.ons.gov.uk'):
        self._apicall = requests.get(API_URL).json()
        self._onssoup = BeautifulSoup(
            requests.get(ONS_URL).content, features = 'html')

        self._count = self._apicall.get('count')
        self._items = self._apicall.get('items')

        self.bulletins = dict()
        self.items = dict()
        self.links = dict()

        self._nlp = spacy.load('en_core_web_sm')
        
    def get_sections(self):
        '''
        from how this is coded it will return a list with entries
        /main_section/sub_section
        so the main_section is [1]
        I then extract the main_section to build my primary dictionary
        with subentries the sub_sections
        '''
        all_sections = [x['href'] for x in self._onssoup.find_all(
            'a', {'class': "primary-nav__child-link"})]
        
        # not sure if the cleanest way
        # iterate over main sections and append the subsection
        # creates the following structure
        # {main_section1 : {subsection1 : [], subsection2: [], ...}, ...}
        self.sections = {}
        for line in all_sections:
            main, sub = line.split('/')[1], line.split('/')[2]
            self.sections.setdefault(main, {})
            self.sections[main].update({sub: []})

    def collect_bulletin(self, max_size = 1000):
        '''
        visits ONS website
        extracts bulletin for all 
        sections found in the main navigation bar:
            businessindustryandtrade, economy, employmentandlabourmarket, 
            peoplepopulationandcommunity
        '''

        # in case you haven't automatically
        # extracted all the sections, I do it for you
        if not hasattr(self, 'sections'):
            self.get_sections()


        print(f'Now collecting publications...')
        for section in tqdm(self.sections.keys()):
            print(f'Working on: "{section}"')
            for subsection in self.sections[section]:
                print(f'    {subsection}')
                url = \
                    'https://www.ons.gov.uk/' + section + '/' + subsection + '/publications' + f'?sortBy=release_date&query=&size={max_size}'
                soup = BeautifulSoup(requests.get(url).content, features = 'html')
                bulletins = [x['data-gtm-uri'] for x in soup.find_all('a', {'data-gtm-uri':True})]
                [self.sections.get(section).get(subsection).append(entry) for entry in bulletins]

    def process_bulletin(self, bulletin_url, log_errors = True):
        '''
        extracts main_points from the bulletin
        if section main_points doesn't exists
        then TODO/ epies kala

        url should be provided in relative format 
        (does not have ons.gov.uk/ at the front)
        that is stored in self.sections[section][subsection][x]
        '''
        if bulletin_url[0] == '/':
            bulletin_url = bulletin_url[1:]
        url = 'https://www.ons.gov.uk/' + bulletin_url
        soup = BeautifulSoup(requests.get(url).content, features= 'html')
        div_mainpoints = soup.find('div', {'id': 'main-points'})
        if div_mainpoints is None:
            if log_errors:
                logging.info(f'Bulletin: {bulletin_url}, MainPoints: None')
                # print('Main Points section not found')
            return
    
        # it seems that some main points have unicode
        # characters that I need to remove
        # for example \xa0 (non-breaking whitespace)
        # might not be the classiest way to do this
        main_points = [x.text.replace(u'\xa0', ' ') \
            for x in div_mainpoints.find_all('p')]

        datasets_soup = BeautifulSoup(requests.get(url + '/relateddata').content, features = 'html')
        related_datasets = [x['data-gtm-uri'] for x in datasets_soup.find_all('a', {'data-gtm-uri':True})]

        # here I use spaCy to check if our main-points
        # have NE that contain DATE or PERCENT
        # arguing that these sentences can easily be used
        date_and_percent, just_dates = self.points_entity_processing(main_points)

        # TODO: this hard sets the dictionary
        # I think there is no issue of overwriting
        # but ideally we would update instead of assigning
        self.bulletins[bulletin_url] = {
            'main-points': main_points,
            'related-datasets': related_datasets,
            'date-and-percent': date_and_percent,
            'just-dates': just_dates
        }
        
    def process_collection(self):
        sections = self.sections.keys()
        for section in tqdm(sections):
            subsections = self.sections.get(section).keys()
            for subsection in tqdm(subsections):
                urls = self.sections.get(section).get(subsection)
                for url in urls:
                    self.process_bulletin(url)

    def points_entity_processing(self, main_points):
        date_and_percent = []
        just_dates = []

        for i, point in enumerate(main_points):

            entities = self._nlp(point).ents
            labels = [x.label_ for x in entities]
            lemmatised = [x.lemma_ for x in entities]
            if 'DATE' and 'PERCENT' in labels:
                date_and_percent.append(i)
            elif 'DATE' in labels:
                just_dates.append(i)
            
        return date_and_percent, just_dates

@attr.s(hash = True)
class Cloze:
    cloze_id = attr.ib()
    point = attr.ib()
    source_text = attr.ib()
    source_start = attr.ib()
    cloze_text = attr.ib()
    answer_text = attr.ib()
    answer_start = attr.ib()
    constituency_parse = attr.ib()
    root_label = attr.ib()
    answer_type = attr.ib()
    question_text = attr.ib()



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