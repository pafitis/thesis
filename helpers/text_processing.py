import numpy as np
import pandas as pd
import logging
import pickle
import spacy

def mainpoints_entity_processing(main_points):
    date_and_percent = []
    just_dates = []

    for i, point in enumerate(main_points):

        entities = nlp(point).ents
        labels = [x.label_ for x in entities]
        lemmatised = [x.lemma_ for x in entities]
        if 'DATE' and 'PERCENT' in labels:
            date_and_percent.append(i)
        elif 'DATE' in labels:
            just_dates.append(i)
        
    return date_and_percent, just_dates
    