import numpy as np

from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from helpers.cloze_generation import generate_clozes_from_point,named_entity_answer_generator as ne_answer_generator, noun_phrase_answer_generator as np_answer_generator, multitoken_mask_answer

import pandas as pd
import numpy as np
import pickle, logging, spacy, sys, os, json

from helpers.classes import Collection
from tqdm import tqdm
from datetime import datetime

import torch

# def run_t5(
#     sentence_list, model, tokenizer, verbose = False):

#     '''
#     pass
#     '''

#     device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     model = model.to(device)

#     answers_given = []

#     for sent in sentence_list:

def check_t5(model, tokenizer, sentence, num_beams = 100, num_return_sequences=1, max_length = 5):

    def replace_mask(
        sentence,
        masks = ['IDENTITYMASK', 'NOUNPHRASEMASK', 'NUMERICMASK', 
            'PLACEMASK', 'TEMPORALMASK', 'THINGMASK']):
        
        x = [sentence.replace(x, "<extra_id_0>") for x in masks if x in sentence]
        
        if len(x):
            return x[0]

    def _filter(output, end_token='<extra_id_1>'):
        # The first token is <unk> (inidex at 0) 
        # and the second token is <extra_id_0> (indexed at 32099)
        _txt = tokenizer.decode(
            output[2:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if end_token in _txt:
            _end_token_index = _txt.index(end_token)
            ans = _txt[:_end_token_index]
            return _result_prefix + _txt[:_end_token_index] + _result_suffix, ans
        else:
            return _result_prefix + _txt + _result_suffix, None

    sent = replace_mask(sentence)
    device = 'cuda:0' if torch.cuda.is_available else 'cpu'
    if sent != None:

        input_ids = tokenizer.encode_plus(
            sent, add_special_tokens = True, return_tensors = 'pt').input_ids.to(device)
        outputs = model.generate(
            input_ids=input_ids, num_beams=num_beams, num_return_sequences=num_return_sequences, max_length = max_length)
        

        _0_index = sent.index('<extra_id_0>')
        _result_prefix = sent[:_0_index]
        _result_suffix = sent[_0_index+12:]  # 12 is the length of <extra_id_0>

        results = list(map(_filter, outputs))
        return results


    else:
        return None

    


def run_t5(df, model, tokenizer, save_results, multi_token = False):

    final_results = []
    entity_set = set()

    for row in tqdm(range(df.shape[0])):
        row_result = []

        if multi_token:
            pass
        else:
            clozes = [c for c in generate_clozes_from_point(
                df['point'][row], ne_answer_generator)]
        
        [entity_set.add((x.answer_text, x.answer_type)) for x in clozes]


        for cloze in clozes:
            if multi_token:
                pass
            else:
                _results = check_t5(model, tokenizer, cloze.cloze_text)
                # this slicing [0] might be problematic in future, i do this because of the list(map(_filter, ...))
                predicted_sentence, predicted_answer = _results[0]

            answer_true = cloze.answer_text
            original_text = cloze.source_text.text
            row_result.append(
                (predicted_answer, predicted_sentence, 
                answer_true, original_text,
                row, cloze.cloze_id)
            )

        final_results.append(row_result)

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
            json.dump(final_results, f)
        with open(
            f'results/{model_name}_{current_time}_entity_set.pickle', 'wb') as f:
            pickle.dump(entity_set, f)
        with open(
            f'results/{model_name}_{current_time}_entity_dictionary.json', 'w') as f:
            json.dump(entities, f)

    return final_results, entity_set, entities


def summarise_t5_results(results, verbose = True):
    '''
    computes EXACT MATCH from T5 results

    expects input from run_t5 function from t5_language_model.py
    results = [row1, row2, ..., rowN]
        row_i = [entry1, entry2, ...]
        entry_i = \
            (answer_given, predicted_sentence, true_answer, true_sentence, row_id, cloze_id)
    '''
    count_correct, count_wrong = 0, 0
    correct_preds, wrong_preds = [], []

    for row in results:
        if len(row):
            for entry in row:
                predicted, truth = entry[0], entry[2]
                if predicted == truth:
                    correct_preds.append(predicted)
                    count_correct += 1
                else:
                    wrong_preds.append((predicted, truth))
                    count_wrong += 1

    if verbose:
        print(f'Total Examples: {count_wrong + count_correct}')
        print(f'Correct: {count_correct}, Incorrect: {count_wrong}')
        print(f'Percentage Correct: {np.round( ((count_correct / (count_correct+ count_wrong) ) * 100), 3)}%')

    return count_correct, count_wrong, correct_preds, wrong_preds


if __name__ == '__main__':
    with open('pickles/dataset_20210625_184837.pkl', 'rb') as f:
        df = pickle.load(f)

    from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration

    model_name = 't5-base'

    config = T5Config.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)
    tokenizer = T5Tokenizer.from_pretrained(model_name, config=config)


    test_df = df[:3]
    results, entity_set, entities = run_t5(
        test_df, model, tokenizer, save_results=False)