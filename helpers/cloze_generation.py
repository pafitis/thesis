from numpy import save
import spacy
import hashlib
from helpers.configs import MIN_ANSWER_CHAR_LEN, MAX_ANSWER_CHAR_LEN,\
    MIN_ANSWER_WORD_LEN, MAX_ANSWER_WORD_LEN, CLOZE_MASKS, MIN_CLOZE_WORD_LEN, MAX_CLOZE_WORD_LEN,\
    MIN_CLOZE_WORDSIZE, MAX_CLOZE_WORDSIZE, MIN_CLOZE_CHAR_LEN, MAX_CLOZE_CHAR_LEN,  \
    MAX_QUESTION_WORDSIZE_THRESHOLD, MAX_PARAGRAPH_WORDSIZE_THRESHOLD, MAX_PARAGRAPH_CHAR_LEN_THRESHOLD, \
    MAX_PARAGRAPH_WORD_LEN_THRESHOLD, MAX_QUESTION_CHAR_LEN_THRESHOLD, MAX_QUESTION_WORD_LEN_THRESHOLD, \
    NOUNPHRASE_LABEL, SPACY_MODEL
from helpers.classes import Cloze

nlp = spacy.load(SPACY_MODEL)

def mask_answer(text, answer_text, answer_start, answer_type):
    '''
    look at noun_phrase_answer_generator and named_entity_answer_generator
    they are responsible for generating the inputs to this function
    
    this function constructs the MASKED sentence,
    you provide indeces of interest and it removes that desired segment and MASKS it
    '''
    before, after = text[:answer_start], text[answer_start + len(answer_text): ]
    return before + CLOZE_MASKS[answer_type] + after

def multitoken_mask_answer(
    text, answer_text, answer_start, tokenizer):
    '''
    constructs MASKED sentence but instead now of just using a single mask, we split the answer into multi-token masks; ie if the answer_text is tokenized into 3 tokens, then the output sentence will contain three masks; works similarly to mark_answer()

    requires tokenizer
    '''
    before, after = text[:answer_start-1], text[answer_start + len(answer_text): ]
    num_tokens = len(tokenizer.tokenize(answer_text))

    return before + (tokenizer.mask_token * num_tokens) + after


def noun_phrase_answer_generator(sentence):
    '''
    returns 3-dim tuple
    (entity, start_pos, label)
    start_pos is useful to know where to place the mask
    '''
    return [
        (noun_phrase.text, noun_phrase.start_char - sentence.start_char, NOUNPHRASE_LABEL) 
        for noun_phrase in sentence.noun_chunks
        ]

def named_entity_answer_generator(sentence, entity_set = None):
    '''
    returns 3-dim tuple
    (entity, start_pos, label)
    start_pos is useful to know where to place the mask
    '''

    return [(ent.text, ent.start_char - sentence.start_char, ent.label_)
        for ent in sentence.ents]

def is_appropriate_cloze(sentence):
    '''returns boolean if sentence is appropriate cloze'''
    good_char_len = MIN_CLOZE_CHAR_LEN < len(sentence) < MAX_CLOZE_CHAR_LEN
    no_links = not (('https://' in sentence) or ('http://' in sentence))
    
    tokens = sentence.split()
    good_num_words = MIN_CLOZE_WORD_LEN <= len(tokens) <= MAX_CLOZE_WORD_LEN
    good_word_lens = all(
        [MIN_CLOZE_WORDSIZE <= len(token) <= MAX_CLOZE_WORDSIZE for token in tokens])

    return good_char_len and no_links and good_word_lens and good_num_words

def is_appropriate_answer(sentence):
    '''returns boolean if sentence is appropriate answer'''
    correct_char_len = MIN_ANSWER_CHAR_LEN <= len(sentence) <= MAX_ANSWER_CHAR_LEN
    correct_word_len = MIN_ANSWER_WORD_LEN <= len(sentence.split()) <= MAX_ANSWER_WORD_LEN
    return correct_char_len and correct_word_len

def get_cloze_id(paragraph_text, sentence_text, answer_text):
    '''
    encodes all paragraph sentence and answer text into one big hash
    this is a unique identifier
    '''
    rep = paragraph_text + sentence_text + answer_text
    return hashlib.sha1(rep.encode()).hexdigest()

def generate_clozes_from_point(point, answer_generator, tokenizer = None):

    clozes = []
    doc = nlp(point)
    entity_set = set()
    multi_token = True if tokenizer else False

    for sentence in doc.sents:
        is_good = is_appropriate_cloze(sentence.text)
        if is_good:
            answers = answer_generator(sentence)
            # answers, entity_set = answer_generator(sentence, entity_set)
            for answer_text, answer_start, answer_type in answers:
                if is_appropriate_answer(answer_text):
                    yield Cloze(
                        cloze_id = get_cloze_id(point, sentence.text, answer_text),
                        source_text = sentence,
                        source_start = sentence.start_char,
                        cloze_text = \
                            multitoken_mask_answer(
                                sentence.text, answer_text, answer_start, tokenizer) 
                            if multi_token else 
                            mask_answer(
                                sentence.text, answer_text, answer_start, answer_type),
                        answer_text = answer_text,
                        answer_start = answer_start,
                        constituency_parse = None,
                        root_label = None,
                        answer_type = answer_type,
                        question_text = None
                    )
    return clozes


if __name__ == '__main__':

    paragraph1 = "We estimate excess winter mortality by comparing the winter months of December to March with the average of the four-month periods before and after"
    paragraph2 = 'The non-financial services sector increased by £25 billion to £744.4 billion'
    paragraphs = [paragraph1, paragraph2]

    import spacy
    nlp = spacy.load('en_core_web_trf')
    answer_generator = named_entity_answer_generator
    clozes = [c for p in paragraphs \
        for c in generate_clozes_from_point(p, answer_generator)]

    # okay this is somewhat hacky
    # here I update a set() which will contain the unique
    # (entity, label) pairs found in our clozes
    # this will be useful (hopefully) later when
    # we want to know more about different labels and their nature
    entity_set = set()
    [entity_set.add((x.answer_text, x.answer_type)) for x in clozes]

    print(clozes)
    print(entity_set)