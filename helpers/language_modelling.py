from generate_cloze import multitoken_mask_answer
import numpy
import torch
from transformers import \
    pipeline, \
    BertForMaskedLM, BertTokenizerFast, RobertaForMaskedLM, RobertaTokenizerFast, ElectraForMaskedLM, ElectraTokenizerFast, AlbertForMaskedLM, AlbertTokenizerFast

from helpers.cloze_generation import \
    generate_clozes_from_point, named_entity_answer_generator as ne_answer_generator, noun_phrase_answer_generator as np_answer_generator

import pandas as pd
import numpy as np
import pickle, logging, spacy, sys, os, json, requests
import matplotlib.pyplot as plt

from helpers.classes import Collection
from tqdm import tqdm
from bs4 import BeautifulSoup
from datetime import datetime


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

def multitoken_prediction(
    sentence, model, tokenizer, top_k_vocab = 1, 
    verbose = False):
    '''
    multitoken prediction. provide a sentence that requires a multi-token answer. the masks will be sequentially predicted, substituting the most confident token and predicting the rest. scales linearly with the number of tokens the answer requires.

    - sentence: input masked sentence; due to RoBERTa please make sure your first mask does not have a space. 
    - model: transformers model.from_pretrained
    - tokenizer: transformer tokenizer.from_pretrained
    - top_k_vocab: experimental, returns k most-confident tokens, defaults to 1
    - verbose: bool, useful to see how the masks were sequentially predicted

    returns sentence with all masks predicted
    '''

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    num_tokens = sentence.count(tokenizer.mask_token)
    if not num_tokens:
        raise ValueError('No masks found in sentence.')

    total_confidence = 1
    answer_given = []

    for _iter in range(num_tokens):
        inputs = tokenizer(sentence, return_tensors = 'pt').to(device)
        outputs = model(**inputs)

        # find where masks are located
        is_masked = torch.where(
            inputs.input_ids == tokenizer.mask_token_id, 1, 0
        )
        masked_idxs = torch.nonzero(is_masked)

        # convert to probabilities
        probabilities = torch.softmax(
            outputs.logits[0, masked_idxs[:, 1]], dim = 1)

        # obtain k most confident token predictions
        mask_confidence, token_ids = torch.topk(probabilities, top_k_vocab)

        # selects the mask index that correspond to the most confident prediction; I am slicing [0] because of top_k_vocab will return the k most confident possible tokens. ultimately top_k_vocab is not used, but I am keeping it here for future work
        most_confident = mask_confidence.argmax(dim = 0)[0].item()
        target_token_idx = token_ids[most_confident][0]
        target_token = tokenizer.decode(target_token_idx)
        
        # log confidence as a proxy of probability
        current_confidence = mask_confidence[most_confident][0].item()
        total_confidence *= current_confidence
        
        # find start and end index of <mask> to be removed
        starting_pos = find_nth_substring(
            sentence, tokenizer.mask_token, most_confident)
        ending_pos = starting_pos + len(tokenizer.mask_token)

        # construct new version of sentence
        # replace mask by predicted token
        sentence = sentence[:starting_pos] + \
            target_token + sentence[ending_pos:]

        answer_given.append((target_token, starting_pos))
        # answer_given.append((target_token, target_token_idx))

        if verbose:
            print(f'Iteration: {_iter}, Predicted Token: {target_token}, Iteration Confidence: {current_confidence}, Total Confidence: {total_confidence}')
            print(f'Sentence: {sentence}')
    final_answer = [x[0] for x in sorted(answer_given, key = lambda x: x[1])]

    return sentence, ''.join(final_answer), total_confidence

def check_model(model, tokenizer, input_sentence):
    '''
    single token answer prediction
    '''

    def replace_mask(
        sentence, 
        masks = ['IDENTITYMASK', 'NOUNPHRASEMASK', 'NUMERICMASK', 
            'PLACEMASK', 'TEMPORALMASK', 'THINGMASK']):

        # somewhat hacky
        # checks if sentence contains any of the masks
        # and replaces it with the appropriate tokenizer.mask_token
        x = [sentence.replace(x, fill_mask.tokenizer.mask_token) for x in masks if x in sentence]
        if len(x):
            return x[0]

    fill_mask = pipeline(
        'fill-mask',
        model = model,
        tokenizer = tokenizer
    )
    sent = replace_mask(input_sentence)
    if sent != None:
        return fill_mask(sent)
    else:
        return None



def run_language_model(
    df, model, tokenizer, save_results, 
    multi_token = True):
    '''
    provide dataset with bulletin points, a transformer model and tokenizer
    generates appropriate clozes (whether you want multi-token masks or not)
    and runs them through the language model. returns confidence, predictions and logs results
    '''

    results = []
    entity_set = set()

    for row in tqdm(range(df.shape[0])):
        row_result = []

        # whether you want multi_tokens to be treated as a single mask or multiple ones, please provide the appropriate boolean. the only difference is that we pass a tokenizer to the cloze generation to handle the tokenization
        if multi_token:
            clozes = [
                c for c in generate_clozes_from_point(
                    df['point'][row], ne_answer_generator, tokenizer)]
        else:
            clozes = [c for c in generate_clozes_from_point(
                df['point'][row], ne_answer_generator)]

        # store entity entries for logging. might come in handy what entities we come across
        [entity_set.add((x.answer_text, x.answer_type)) for x in clozes]

        # go through each cloze and generate predictions
        for cloze in clozes:
            # similarly, for single token masks we utilise transformers.pipeline functionality, where for multi_token ones we had to write our own. functionality is largely the same
            # returns the predicted answer with the corresponding confidence
            if multi_token:
                _, answer_given, confidence = multitoken_prediction(
                    cloze.cloze_text, model, tokenizer, top_k_vocab = 1, verbose = False
                )
            else:
                result = check_model(model, tokenizer, cloze.cloze_text)
            
                answer_given = ''.join(result[0].get('token_str').split(' '))
                confidence = result[0].get('score')

            # ground truth
            answer_true = cloze.answer_text

            # saves prediction, confidence score, truth, dataframe row, cloze id
            row_result.append(
                (answer_given, confidence, answer_true, row, cloze.cloze_id))

        results.append(row_result)

    if save_results:
        model_name = model._get_name()
        current_time = str(datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        with open(
            f'results/{model_name}_{current_time}_results.json', 'w') as f:
            json.dump(results, f)
        with open(
            f'results/{model_name}_{current_time}_entity_set.pickle', 'wb') as f:
            pickle.dump(entity_set, f)

        #  here I am saving the entities in a dictionary 
        #  with keys being each different entity category 
        #  such as MONEY, PERCENT and so on with values the unique terms found in our data

        categories = [x[1] for x in list(entity_set)]
        # construct keys
        entities = dict()
        entities = {f'{x}':[] for x in categories if x not in entities}
        # append only unique values
        [entities.get(x[1]).append(x[0]) for x in entity_set if x[0] not in entities.get(x[1])]

        with open(
            f'results/{model_name}_{current_time}_entity_dictionary.json', 'w') as f:
            json.dump(entities, f)

    return results, entity_set, entities



if __name__ == '__main__':
    with open('pickles/dataset_20210625_184837.pkl', 'rb') as f:
        df = pickle.load(f)

    from transformers import RobertaTokenizer, RobertaForMaskedLM
    
    model = RobertaForMaskedLM.from_pretrained('roberta-base')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    test_df = df[:3]
    results, entity_set, entities = run_language_model(test_df, model, tokenizer, True)