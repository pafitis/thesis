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

    from transformers import TapasTokenizer, TapasForQuestionAnswering, TapasForMaskedLM, TapasConfig, AutoTokenizer, AutoModel
    import pandas as pd

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


    dir_name = '/Users/pafitis/dev/comp0087/thesis/models/tapas_wtq_wikisql_sqa_inter_masklm_large_reset_pt'

    # config = TapasConfig.from_pretrained(
    #     f'{dir_name}',from_pt=True)
    # model = TapasForMaskedLM.from_pretrained(
    #     f'{dir_name}', config=config)
    # tokenizer=TapasTokenizer.from_pretrained(
    #     f'{dir_name}', from_pt=True)
    # model = TapasForMaskedLM.from_pretrained(
    #     f'google/tapas-base')
    # tokenizer=TapasTokenizer.from_pretrained(
    #     f'google/tapas-base', from_pt=True)
    tokenizer = AutoTokenizer.from_pretrained(
        "google/tapas-large")
    model = AutoModel.from_pretrained(
        "google/tapas-large")

    data = {
        'Actors': ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
        'Age': ["56", "45", "59"],
        'Number of movies': ["87", "53", "69"]
        }

    queries = [
        "George Clooney played in [MASK] movies?", 
        "Brad Pitt is [MASK] old?"
        ]
    
    table = pd.DataFrame.from_dict(data)

    inputs1 = tokenizer(
        table=table, queries=queries[0],
        padding='max_length', return_tensors='pt',
        truncation=True)

    inputs2 = tokenizer(
        table=table, queries=queries[1],
        padding='max_length', return_tensors='pt',
        truncation=True)
    
    for inputs in [inputs1, inputs2]:

        outputs = model(**inputs)

        ######
        sentence = queries[0]
        verbose = True
        _iter = 0
        sequence_confidence = 0
        sequence_confidences = []

        answer_given = []

        top_k_vocab = 1
        # find where masks are located
        is_masked = torch.where(
            inputs.input_ids == tokenizer.mask_token_id, 1, 0
            )
        masked_idxs = torch.nonzero(is_masked)

        # convert to probabilities
        probabilities = torch.softmax(
            outputs.last_hidden_state[0, masked_idxs[:, 1]],
            dim = 1
            )
        # probabilities = torch.softmax(
        #     outputs.logits[0, masked_idxs[:, 1]],
        #     dim = 1
        #     )
        logprobs = torch.log(probabilities)

        # obtain k most confident token predictions, work on logprobs to avoid underflow
        mask_confidence, token_ids = torch.topk(logprobs, top_k_vocab)

        # selects the mask index that correspond to the most confident prediction; I am slicing [0] because of top_k_vocab will return the k most confident possible tokens. ultimately top_k_vocab is not used, but I am keeping it here for future work
        most_confident = mask_confidence.argmax(dim = 0)[0].item()
        target_token_idx = token_ids[most_confident][0]
        target_token = ''.join(tokenizer.decode(target_token_idx).split(' '))
        print(target_token)
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
            
# if __name__ == '__main__':

#     from transformers import TapasConfig,TapasTokenizer,TapasForMaskedLM
#     from transformers import pipeline
#     import pandas as pd
#     import numpy as np
#     import torch
#     import sys

#     config = TapasConfig.from_pretrained(
#         'google/tapas-base-finetuned-wtq',from_pt=True)
#     model = TapasForMaskedLM.from_pretrained(
#         'google/tapas-base-finetuned-wtq', config=config)
#     tokenizer=TapasTokenizer.from_pretrained(
#         "google/tapas-base-finetuned-wtq", from_pt=True)

#     # outdir = "tmp"

#     # model.save_pretrained(outdir)
#     # tokenizer.save_pretrained(outdir)
#     # config.save_pretrained(outdir)


#     nlp = pipeline(task="fill-mask",framework="pt",model=model, tokenizer=tokenizer)
#     #nlp = pipeline(task="table-question-answering")


#     data= {
#         "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
#         "age": ["56", "45", "59"],
#         "number of movies": ["87", "53", "69"],
#         "date of birth": ["7 february 1967", "10 june 1996", "28 november 1967"]
#     }

#     table = pd.DataFrame.from_dict(data)

#     queries=[
#         f"The number of movies Brad Pitt acted in is {tokenizer.mask_token}",
#         f"Leonardo di caprio's age is {tokenizer.mask_token}"]

#     test = nlp(queries, table = pd.DataFrame(table))