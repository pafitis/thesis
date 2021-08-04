if __name__ == '__main__':
    import torch
    import pandas as pd
    import numpy as np
    import pickle, logging, spacy, sys, os, json, requests
    import matplotlib.pyplot as plt

    from helpers.classes import Collection
    from tqdm import tqdm
    from bs4 import BeautifulSoup
    from datetime import datetime

    from helpers.cloze_generation import generate_clozes_from_point, named_entity_answer_generator as ne_answer_generator, noun_phrase_answer_generator as np_answer_generator

    from helpers.table_processing import preprocess_table

    from helpers.language_modelling import run_language_model, summarise_results

    df = pd.read_pickle('pickles/dataset_20210625_184837.pkl')
    clozes_df = pd.read_json('pickles/clozes_20210715_212425.json')

    from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration


    def replace_mask(
        sentence, 
        masks = ['IDENTITYMASK', 'NOUNPHRASEMASK', 'NUMERICMASK', 
            'PLACEMASK', 'TEMPORALMASK', 'THINGMASK']):

        # somewhat hacky
        # checks if sentence contains any of the masks
        # and replaces it with the appropriate tokenizer.mask_token
        x = [sentence.replace(x, tokenizer.mask_token) \
            for x in masks if x in sentence]
        if len(x):
            return x[0]


    def find_nth_substring(sentence, substring, n):
        '''
        used internally in multitoken_prediction to find the n-th occurence of a specific substring in a sentence
        returns the starting index of substring
        '''
        start = sentence.find(substring)
        while start >= 0 and n > 1:
            start = sentence.find(substring, start+len(substring))
            n -= 1
        return start


    # model_name = 't5-base'
    model_name = 't5-large'

    config = T5Config.from_pretrained(model_name)
    config.max_length, config.min_length = 100, 50

    model = T5ForConditionalGeneration.from_pretrained(model_name, config = config)
    tokenizer = T5Tokenizer.from_pretrained(model_name, config = config)

    print(model.config.min_length)
    print(model.config.max_length)
    # text = "Pavlos really <extra_id_0> Andrestinos."

    # encoded = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')
    # input_ids = encoded['input_ids']

    # outputs = model.generate(
    #     input_ids=input_ids, num_beams=10, 
    #     num_return_sequences=5)

    # _0_index = text.index('<extra_id_0>')
    # _result_prefix = text[:_0_index]
    # _result_suffix = text[_0_index+12:]  # 12 is the length of <extra_id_0>

    # def _filter(output, end_token='<extra_id_1>'):
    #     # The first token is <unk> (inidex at 0) and the second token is <extra_id_0> (indexed at 32099)
    #     _txt = tokenizer.decode(
    #         output[2:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    #     if end_token in _txt:
    #         _end_token_index = _txt.index(end_token)
    #         return _result_prefix + _txt[:_end_token_index] + _result_suffix
    #     else:
    #         return _result_prefix + _txt + _result_suffix

    # results = list(map(_filter, outputs))
    # print(results)
