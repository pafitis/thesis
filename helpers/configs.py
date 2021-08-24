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



# EXCEL PROCESSING

# multiplier that determines which rows to remove
# threshold * median < X --> remove
ROW_NAN_THRESHOLD = 1.5

# if columns should be forward-filled to impute
# nan values. usefull for columns that are labels for groups
IMPUTE_COL_NANS = True
IMPUTE_COL_THRESHOLD = 0.9 # threshold to determine if to be imputed

# hacky way to understand which entries are actual column headers
# we drop until we find non-"Unnamed" entries
DROP_UNNAMED = True

# this fills all NAN values in rows to '-' to match empty entries in ONS data
FILL_NA = True

# backfills headers after processing the column names; this is as sometimes there are headers at different levels. this tries to fix that
BACKFILL_HEADERS = True

# experimental; fills forward empty (" ") or NaN values in the first column in an attempt to fix tables that have multi-level headers
IMPUTE_FIRST_COL_EMPTY = True

# experimental; fill fowrard empty (" ") or NaN values in all columns
IMPUTE_ALL_COL_EMPTY = True

# the number of characters we allow for lost column headers, anything more than this is replaced with an empty string
LOST_COL_LEN_THRESHOLD = 100