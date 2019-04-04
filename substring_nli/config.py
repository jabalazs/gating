import os

DATA_PATH = "data/"
CACHE_PATH = ".cache"
TRAINED_MODELS_PATH = os.path.join(DATA_PATH, "trained_models")
LOG_PATH = "log/"

# EMBEDDINGS
EMBEDDINGS_DIR = os.path.join(DATA_PATH, "word_embeddings")
PREPROCESSED_DATA_PATH = os.path.join(DATA_PATH, "preprocessed")
SENNA_EMB_PATH = os.path.join(EMBEDDINGS_DIR, "senna.txt")
GLOVE_EMB_PATH = os.path.join(EMBEDDINGS_DIR, "glove.840B.300d.txt")
FASTTEXT_EMB_PATH = os.path.join(EMBEDDINGS_DIR, "wiki.en.vec")

# CORPORA
CORPORA_DIR = os.path.join(DATA_PATH, "corpora")

MULTINLI_CORPUS_DIR = os.path.join(CORPORA_DIR, "multinli_0.9")

MULTINLI_TRAIN_PATH = os.path.join(
    MULTINLI_CORPUS_DIR, "multinli_0.9_train.jsonl"
)
MULTINLI_DEV_MATCHED_PATH = os.path.join(
    MULTINLI_CORPUS_DIR, "multinli_0.9_dev_matched.jsonl"
)
MULTINLI_DEV_MISMATCHED_PATH = os.path.join(
    MULTINLI_CORPUS_DIR, "multinli_0.9_dev_mismatched.jsonl"
)

MULTINLI_TRAIN_PREPROCESSED_PATH = os.path.join(
    PREPROCESSED_DATA_PATH, "multinli_0.9_train_preprocessed.jsonl"
)
MULTINLI_DEV_MATCHED_PREPROCESSED_PATH = os.path.join(
    PREPROCESSED_DATA_PATH, "multinli_0.9_dev_matched_preprocessed.jsonl"
)
MULTINLI_DEV_MISMATCHED_PREPROCESSED_PATH = os.path.join(
    PREPROCESSED_DATA_PATH, "multinli_0.9_dev_mismatched_preprocessed.jsonl"
)

MULTINLI_TEST_PATH = os.path.join(MULTINLI_CORPUS_DIR, "multinli_0.9_test.jsonl")

MULTINLI_TOKEN_DICT_PICKLE_PATH = os.path.join(
    CACHE_PATH, "multinli_train_token_dict.pkl"
)
MULTINLI_CHAR_DICT_PICKLE_PATH = os.path.join(
    CACHE_PATH, "multinli_train_char_dict.pkl"
)

SNLI_CORPUS_DIR = os.path.join(CORPORA_DIR, "snli_1.0")
SNLI_TRAIN_PATH = os.path.join(SNLI_CORPUS_DIR, "snli_1.0_train.jsonl")
SNLI_TRAIN_PREPROCESSED_PATH = os.path.join(
    PREPROCESSED_DATA_PATH, "snli_1.0_train_preprocessed.jsonl"
)
SNLI_DEV_PATH = os.path.join(SNLI_CORPUS_DIR, "snli_1.0_dev.jsonl")
SNLI_DEV_PREPROCESSED_PATH = os.path.join(
    PREPROCESSED_DATA_PATH, "snli_1.0_dev_preprocessed.jsonl"
)
SNLI_TEST_PATH = os.path.join(SNLI_CORPUS_DIR, "snli_1.0_test.jsonl")
SNLI_TEST_PREPROCESSED_PATH = os.path.join(
    PREPROCESSED_DATA_PATH, "snli_1.0_test_preprocessed.jsonl"
)

SNLI_TOKEN_DICT_PICKLE_PATH = os.path.join(
    CACHE_PATH, "snli_train_token_dict.pkl"
)
SNLI_CHAR_DICT_PICKLE_PATH = os.path.join(CACHE_PATH, "snli_train_char_dict.pkl")


# MAPPINGS
embedding_dict = {
    "senna": SENNA_EMB_PATH,
    "glove": GLOVE_EMB_PATH,
    "fasttext": FASTTEXT_EMB_PATH,
}

corpora_dict = {
    "multinli": {
        "train": MULTINLI_TRAIN_PREPROCESSED_PATH,
        "dev_matched": MULTINLI_DEV_MATCHED_PREPROCESSED_PATH,
        "dev_mismatched": MULTINLI_DEV_MISMATCHED_PREPROCESSED_PATH,
        "test": MULTINLI_TEST_PATH,
    },
    "snli": {
        "train": SNLI_TRAIN_PREPROCESSED_PATH,
        "dev": SNLI_DEV_PREPROCESSED_PATH,
        "test": SNLI_TEST_PREPROCESSED_PATH,
    },
}

WRITE_MODES = {"none": None, "file": "FILE", "db": "DATABASE", "both": "BOTH"}

PAD_ID = 0
UNK_ID = 1
NUM_ID = 2
URL_ID = 3
# SOS_ID = 4
# EOS_ID = 5

PAD_TOKEN = "__PAD__"
UNK_TOKEN = "__UNK__"
NUM_TOKEN = "__NUM__"
URL_TOKEN = "__URL__"
# SOS_TOKEN = '__SOS__'
# EOS_TOKEN = '__EOS__'

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, NUM_TOKEN, URL_TOKEN]

UNK_CHAR_ID = 0
UNK_CHAR_TOKEN = "あ"

LABEL2ID = {"neutral": 0, "contradiction": 1, "entailment": 2}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}

LABELS = ["neutral", "contradiction", "entailment"]

# DATABASE PARAMETERS
_DB_NAME = "runs.db"

DATABASE_CONNECTION_STRING = "sqlite:///" + os.path.join(
    TRAINED_MODELS_PATH, _DB_NAME
)


JSON_KEYFILE_PATH = "experiments-database-8ee4da525610.json"

try:
    SERVER_NAME = open("server_name", "r").read().strip()
except FileNotFoundError:
    pass

SPREADSHEET_NAME = "substring_nli-03"
