import os

HERE = os.path.dirname(os.path.realpath(__file__))
SEED = 96

# spaCy
SPACY_MODEL = 'en_core_web_trf'

## Cloze thresholds
MIN_CLOZE_WORD_LEN = 5
MAX_CLOZE_WORD_LEN = 40
MIN_CLOZE_WORDSIZE = 1
MAX_CLOZE_WORDSIZE = 20
MIN_CLOZE_CHAR_LEN = 30
MAX_CLOZE_CHAR_LEN = 300
MIN_ANSWER_WORD_LEN = 1
MAX_ANSWER_WORD_LEN = 20
MIN_ANSWER_CHAR_LEN = 3
MAX_ANSWER_CHAR_LEN = 50

## Maximum-size thresholds
# characters
MAX_PARAGRAPH_CHAR_LEN_THRESHOLD = 2000
MAX_QUESTION_CHAR_LEN_THRESHOLD = 200
## words
MAX_PARAGRAPH_WORD_LEN_THRESHOLD = 400
MAX_QUESTION_WORD_LEN_THRESHOLD = 40
## word-character
MAX_PARAGRAPH_WORDSIZE_THRESHOLD = 20
MAX_QUESTION_WORDSIZE_THRESHOLD = 20



# CLOZE MASKS:
NOUNPHRASE_LABEL = 'NOUNPHRASE'
CLOZE_MASKS = {
    'PERSON': 'IDENTITYMASK',
    'NORP': 'IDENTITYMASK',
    'FAC': 'PLACEMASK',
    'ORG': 'IDENTITYMASK',
    'GPE': 'PLACEMASK',
    'LOC': 'PLACEMASK',
    'PRODUCT': 'THINGMASK',
    'EVENT': 'THINGMASK',
    'WORKOFART': 'THINGMASK',
    'WORK_OF_ART': 'THINGMASK',
    'LAW': 'THINGMASK',
    'LANGUAGE': 'THINGMASK',
    'DATE': 'TEMPORALMASK',
    'TIME': 'TEMPORALMASK',
    'PERCENT': 'NUMERICMASK',
    'MONEY': 'NUMERICMASK',
    'QUANTITY': 'NUMERICMASK',
    'ORDINAL': 'NUMERICMASK',
    'CARDINAL': 'NUMERICMASK',
    NOUNPHRASE_LABEL: 'NOUNPHRASEMASK'
}

HEURISTIC_CLOZE_TYPE_QUESTION_MAP = {
    'PERSON': ['Who', ],
    'NORP': ['Who', ],
    'FAC': ['Where', ],
    'ORG': ['Who', ],
    'GPE': ['Where', ],
    'LOC': ['Where', ],
    'PRODUCT': ['What', ],
    'EVENT': ['What', ],
    'WORKOFART': ['What', ],
    'WORK_OF_ART': ['What', ],
    'LAW': ['What', ],
    'LANGUAGE': ['What', ],
    'DATE': ['When', ],
    'TIME': ['When', ],
    'PERCENT': ['How much', 'How many'],
    'MONEY': ['How much', 'How many'],
    'QUANTITY': ['How much', 'How many'],
    'ORDINAL': ['How much', 'How many'],
    'CARDINAL': ['How much', 'How many'],
}