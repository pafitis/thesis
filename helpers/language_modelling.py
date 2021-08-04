
import torch
from transformers import \
    pipeline, \
    BertForMaskedLM, BertTokenizerFast, RobertaForMaskedLM, RobertaTokenizerFast, ElectraForMaskedLM, ElectraTokenizerFast, AlbertForMaskedLM, AlbertTokenizerFast

from helpers.cloze_generation import generate_clozes_from_point,named_entity_answer_generator as ne_answer_generator, noun_phrase_answer_generator as np_answer_generator, multitoken_mask_answer

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

def compute_perplexity(input_list, is_log = True):
    if is_log:
        input_list = np.exp(input_list)
    product = np.prod(1 / input_list)
    length = len(input_list)

    return product ** (1/length)

def multitoken_prediction(
    sentence_list, model, tokenizer, top_k_vocab = 1, 
    verbose = False, confidence_normalizer = 'token-length'):
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
    
    # we do not know the actual number of tokens the answer is, hence we check up to 5 length answers
    all_answers = []
    for current_mask_length in range(len(sentence_list)):
        sentence = sentence_list[current_mask_length]

        num_tokens = sentence.count(tokenizer.mask_token)
        if not num_tokens:
            raise ValueError('No masks found in sentence.')

        sequence_confidence = 0
        sequence_confidences = []

        answer_given = []

        # sequentially predict each mask token
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
            logprobs = torch.log(probabilities)

            # obtain k most confident token predictions, work on logprobs to avoid underflow
            mask_confidence, token_ids = torch.topk(logprobs, top_k_vocab)

            # selects the mask index that correspond to the most confident prediction; I am slicing [0] because of top_k_vocab will return the k most confident possible tokens. ultimately top_k_vocab is not used, but I am keeping it here for future work
            most_confident = mask_confidence.argmax(dim = 0)[0].item()
            target_token_idx = token_ids[most_confident][0]
            target_token = tokenizer.decode(target_token_idx)
            
            # confidence as a proxy of probability
            token_confidence = mask_confidence[most_confident][0].item()
            
            # add logprobabilities to obtain sequence probability
            sequence_confidence += token_confidence
            sequence_confidences.append(token_confidence)
            
            # find start and end index of <mask> to be removed
            starting_pos = find_nth_substring(
                sentence, tokenizer.mask_token, most_confident)
            ending_pos = starting_pos + len(tokenizer.mask_token)

            # construct new version of sentence
            # replace mask by predicted token
            sentence = sentence[:starting_pos] + \
                target_token + sentence[ending_pos:]

            # answer_given = [ (token, position), ... ]
            answer_given.append((target_token, starting_pos))

            if verbose:
                print(f'Iteration: {_iter}, Predicted Token: {target_token}, Iteration Confidence: {token_confidence}, Confidence (sequence): {sequence_confidence}')
                print(f'Sentence: {sentence}')

        # construct answer in correct order, as we predict tokens in random positions
        final_answer = [x[0] for x in sorted(answer_given, key = lambda x: x[1])]
        # returns list of tokens, combine them together and make sure we remove the starting whitespace
        num_tokens = len(answer_given)
        final_answer = ''.join(final_answer)
        final_answer = final_answer[1:] if final_answer[0] == ' ' else final_answer
        
        # compute perplexity
        # minimizing perplexity is equivalent to maximizing the test set probability according to the language model.
        perplexity = compute_perplexity(sequence_confidences, is_log=True)

        all_answers.append(
            (final_answer, sequence_confidence, perplexity, sequence_confidences, num_tokens))

    # if confidence_normalizer == 'token-length':
    #     confidences = [
    #         _sum_confidence / _num_tokens for \
    #              _, _sum_confidence, _, _num_tokens in all_answers]
    # if confidence_normalizer == 'char-length':
    #     confidences = [x[1] / len(x[0]) for x in all_answers]
    
    most_confident = np.argmax([x[1] for x in all_answers])
    less_perplex = np.argmin([x[2] for x in all_answers])
    
    return all_answers[most_confident], all_answers[less_perplex]


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
        x = [sentence.replace(x, fill_mask.tokenizer.mask_token) \
            for x in masks if x in sentence]
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
    multi_token = True, filter_labels = []):
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
            # generate_clozes_from_point (multitoken) will assign a list for cloze.cloze_text. this is because we don't know the exact number of tokens to explore. 
            # we check up to answers with 5 tokens. hence cloze.cloze_text is usually a 5 length list of sentences of increasing number of masks
            # clozes = [cloze1, cloze2, cloze3, ...]
            # cloze_i = [cloze_i_1, cloze_i_2, ..., cloze_i_5]
            clozes = [
                c for c in generate_clozes_from_point(
                    df['point'][row], ne_answer_generator, tokenizer, filter_labels)]
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
                answer_confidence, answer_perplexity = multitoken_prediction(
                    cloze.cloze_text, model, tokenizer, 
                    top_k_vocab = 1, verbose = False
                )
            else:
                result = check_model(model, tokenizer, cloze.cloze_text)
            
                answer_given = ''.join(result[0].get('token_str').split(' '))
                confidence = result[0].get('score')

            # ground truth
            answer_true = cloze.answer_text

            # saves prediction, confidence score, truth, dataframe row, cloze id
            row_result.append(
                (answer_confidence, answer_perplexity, answer_true, row, cloze.cloze_id)
            )
            # row_result.append(
            #     (answer_given, confidence, answer_true, row, cloze.cloze_id))

        results.append(row_result)

    #  here I am saving the entities in a dictionary 
    #  with keys being each different entity category 
    #  such as MONEY, PERCENT and so on with values the unique terms found in our data

    categories = [x[1] for x in list(entity_set)]
    # construct keys
    entities = dict()
    entities = {f'{x}':[] for x in categories if x not in entities}
    # append only unique values
    [entities.get(x[1]).append(x[0]) \
        for x in entity_set if x[0] not in entities.get(x[1])]
    
    if save_results:
        model_name = model._get_name()
        current_time = str(datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        with open(
            f'results/{model_name}_{current_time}_results.json', 'w') as f:
            json.dump(results, f)
        with open(
            f'results/{model_name}_{current_time}_entity_set.pickle', 'wb') as f:
            pickle.dump(entity_set, f)
        with open(
            f'results/{model_name}_{current_time}_entity_dictionary.json', 'w') as f:
            json.dump(entities, f)

    return results, entity_set, entities

def summarise_results(results, includes_perplexity = False, verbose = True):
    '''
    wrapper function that uses input from run_language_model.py to determine the number of correct predictions, incorrect predictions and other summary metrics.
    
    we introduced perplexity measures so this handles that too
    input results needs to be formatted as

    results = [row1, row2, ..., row2015]
        row1 = [entry_1, entry_2, ..., entry_N]
        if includes_perplexity:
            entry_i = [ [confidence_results], [perplexity_results], truth, row_id, cloze_id]
                confidence_results = \
                    [confidence_answer, confidence, perplexity, seq_confidences, token_length]
                perplexity_results = \
                     [perplexity_answer, confidence, perplexity, seq_confidences, token_length]
        else:
            entry_i = [ confidence_answer, confidence, truth, row_id, cloze_id]
    '''

    if includes_perplexity:
        perp_count_correct, perp_count_wrong = 0, 0
        conf_count_correct, conf_count_wrong = 0, 0

        perp_correct_preds, perp_wrong_preds = [], []
        conf_correct_preds, conf_wrong_preds = [], []

    count_correct, count_wrong = 0, 0
    correct_preds, wrong_preds = [], []

    for row in results:
        if len(row):
            for entry in row:
                if includes_perplexity:
                    confidence_results = entry[0]
                    perplexity_results = entry[1]

                    truth = entry[2]

                    conf_answer = confidence_results[0]
                    perp_answer = perplexity_results[0]
                    # remove whitespace
                    for ans in [conf_answer, perp_answer]:
                        if ans[0] == ' ':
                            ans = ans[1:]
                    
                    if (conf_answer or perp_answer) == truth:
                        count_correct += 1
                        if conf_answer == truth:
                            conf_count_correct += 1
                            conf_correct_preds.append(conf_answer)
                        else:
                            perp_count_correct += 1
                            perp_correct_preds.append(perp_answer)

                    else:
                        count_wrong += 1
                        conf_count_wrong += 1
                        perp_count_wrong += 1

                        perp_wrong_preds.append((perp_answer, truth))
                        conf_wrong_preds.append((conf_answer, truth))

                else:    
                    # the way the mask LM works it includes a whitespace
                    # whereas our mask does not; this fixes that in the accuracy metrics
                    if entry[0][0] == ' ':
                        entry[0] = entry[0][1:]
                    if entry[0] == entry[2]:
                        count_correct += 1
                        correct_preds.append(entry[0])
                    else:
                        count_wrong += 1
                        wrong_preds.append((entry[0], entry[2]))
    if verbose:
        print(f'Total Examples: {count_wrong + count_correct}')
        print(f'Correct: {count_correct}, Incorrect: {count_wrong}')
        print(f'Percentage Correct: {np.round( ((count_correct / (count_correct+ count_wrong) ) * 100), 3)}%')
        if includes_perplexity:
            print(f'Confidence-based Answers: {np.round( ((conf_count_correct / (conf_count_correct + conf_count_wrong) ) * 100), 3)}')
            print(f'Perplexity-based Answers: {np.round( ((perp_count_correct / (perp_count_correct + perp_count_wrong) ) * 100), 3)}')

    if includes_perplexity:
        return count_correct, count_wrong, correct_preds, wrong_preds, \
            conf_count_correct, conf_count_wrong, conf_correct_preds, conf_wrong_preds, perp_count_correct, perp_count_wrong, perp_correct_preds, perp_wrong_preds
    else:
        return count_correct, count_wrong, correct_preds, wrong_preds

if __name__ == '__main__':
    with open('pickles/dataset_20210625_184837.pkl', 'rb') as f:
        df = pickle.load(f)

    from transformers import RobertaTokenizer, RobertaForMaskedLM
    
    model = RobertaForMaskedLM.from_pretrained('roberta-base')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    test_df = df[:3]
    results, entity_set, entities = run_language_model(test_df, model, tokenizer, save_results=False)