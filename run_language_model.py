import pandas as pd
import numpy as np
import pickle, logging, spacy, sys, os, json, requests
import matplotlib.pyplot as plt

from helpers.classes import Collection
from tqdm import tqdm
from bs4 import BeautifulSoup
from datetime import datetime
from time import time
from transformers import RobertaTokenizer, RobertaForMaskedLM

import torch

from helpers.cloze_generation import \
    generate_clozes_from_point, named_entity_answer_generator as ne_answer_generator, noun_phrase_answer_generator as np_answer_generator

from helpers.language_modelling import \
    find_nth_substring, multitoken_prediction, check_model, run_language_model

with open('pickles/dataset_20210625_184837.pkl', 'rb') as f:
    df = pickle.load(f)


print('Initialising language model...')
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = RobertaForMaskedLM.from_pretrained('roberta-base').to(device)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# test_df = df[:3]

start_time = time()
print('Running language model...')
results, entity_set, entities = run_language_model(df, model, tokenizer, True)

end_time = time()

print('Done!')
print(f'Time took: {end_time - start_time}')