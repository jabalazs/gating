# import os
import re
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize.treebank import TreebankWordTokenizer

from ..preprocessing import preprocess
from .. import config
from ..utils.io import load_or_create


def postprocess_vocab(vocab: dict) -> dict:
    # WARNING: make sure these 4 ids are contiguous! eg 0,1,2,3
    new_vocab = {
        config.PAD_TOKEN: config.PAD_ID,
        config.UNK_TOKEN: config.UNK_ID,
        config.NUM_TOKEN: config.NUM_ID,
        config.URL_TOKEN: config.URL_ID,
    }

    curr_id = len(new_vocab)
    # matches any single quotes (once or more times) in front of a token
    # will not match if token is 'nt or 't
    # This should be run after tokenization
    exp = re.compile(r"^('+)(?!(nt)|t)+")
    for key, value in vocab.items():
        postprocessed = exp.sub(r"", key)
        if postprocessed not in new_vocab.keys():
            new_vocab[postprocessed] = curr_id
            curr_id += 1

    return new_vocab


def build_word_lang(sents, min_freq_threshold=0):

    treebank_tokenizer = TreebankWordTokenizer()

    word_vectorizer = CountVectorizer(
        preprocessor=preprocess,
        tokenizer=treebank_tokenizer.tokenize,
        lowercase=True,
    )

    term_doc_matrix = word_vectorizer.fit_transform(sents)

    word_freqs = term_doc_matrix.sum(axis=0).A1
    words = word_vectorizer.get_feature_names()

    word_counts = Counter(dict(zip(words, word_freqs)))

    vocab = {}
    curr_id = 0
    for word in words:
        if word_counts[word] < min_freq_threshold:
            # If too few ocurrences of token, we don't add it to the vocabulary
            # this will make it an unk when transforming tokens to ids
            continue
        if word not in vocab.keys():
            vocab[word] = curr_id
            curr_id += 1

    token2id = postprocess_vocab(vocab)
    # token2id = vocab
    id2token = {value: key for key, value in token2id.items()}

    return {
        "counts": word_counts,
        "token2id": token2id,
        "vectorizer": word_vectorizer,
    }


def build_char_lang(sents):

    char_vectorizer = CountVectorizer(analyzer="char", lowercase=False)
    char_term_doc_matrix = char_vectorizer.fit_transform(sents)
    char_freqs = char_term_doc_matrix.sum(axis=0).A1
    chars = char_vectorizer.get_feature_names()

    char_counts = Counter(dict(zip(chars, char_freqs)))
    char_vocab = {config.UNK_CHAR_TOKEN: config.UNK_CHAR_ID}
    curr_id = len(char_vocab)
    for char in chars:
        if char not in char_vocab.keys():
            char_vocab[char] = curr_id
            curr_id += 1
    return {"counts": char_counts, "char2id": char_vocab}


class Lang(object):

    MODES = ["multinli", "snli"]

    def __init__(
        self,
        sents,
        mode="multinli",
        min_freq_threshold=0,
        force_reload=False,
        token_dict_pickle_path=None,
        char_dict_pickle_path=None,
    ):

        # FIXME: If we only pass one pickle path we'll get an error;
        # we either need to pass both or None
        if (token_dict_pickle_path is None) and (char_dict_pickle_path is None):
            if mode == "multinli":
                token_dict_pickle_path = config.MULTINLI_TOKEN_DICT_PICKLE_PATH
                char_dict_pickle_path = config.MULTINLI_CHAR_DICT_PICKLE_PATH
            elif mode == "snli":
                token_dict_pickle_path = config.SNLI_TOKEN_DICT_PICKLE_PATH
                char_dict_pickle_path = config.SNLI_CHAR_DICT_PICKLE_PATH
            else:
                raise RuntimeError(
                    f"Mode not recognized, try with one of {self.MODES}"
                )

        token_dict = load_or_create(
            token_dict_pickle_path,
            build_word_lang,
            sents,
            min_freq_threshold=min_freq_threshold,
            force_reload=force_reload,
        )

        self.token_counts = token_dict["counts"]
        self.token2id = token_dict["token2id"]
        self.vectorizer = token_dict["vectorizer"]
        self.analyzer = self.vectorizer.build_analyzer()

        char_dict = load_or_create(
            char_dict_pickle_path,
            build_char_lang,
            sents,
            force_reload=force_reload,
        )

        self.char_counts = char_dict["counts"]
        self.char2id = char_dict["char2id"]

    def sent2ids(self, sent, ignore_period=True, append_EOS=False):
        if not isinstance(sent, str):
            raise TypeError(f"Input shout be a str but got {type(sent)} instead.")
        ids = []
        for token in self.analyzer(sent):
            if token == "." and ignore_period:
                continue
            try:
                ids.append(self.token2id[token])
            except KeyError:
                ids.append(config.UNK_ID)
        if append_EOS:
            ids.append(config.EOS_ID)
        return ids

    def sents2ids(self, sents, ignore_period=True, append_EOS=False):
        if not isinstance(sents, list):
            raise TypeError(f"Expected list but got {type(sents)} instead.")
        id_sents = []
        for sent in sents:
            id_sents.append(
                self.sent2ids(
                    sent, ignore_period=ignore_period, append_EOS=append_EOS
                )
            )
        return id_sents

    def token2char_ids(self, token):
        if not isinstance(token, str):
            raise TypeError(
                f"Input shout be a str but got {type(token)} instead."
            )

        if token in (config.UNK_TOKEN, config.NUM_TOKEN, config.URL_TOKEN):
            return [config.UNK_CHAR_ID]

        char_ids = []
        for char in token:
            try:
                char_ids.append(self.char2id[char])
            except KeyError:
                char_ids.append(config.UNK_CHAR_ID)
        return char_ids

    def sent2char_ids(self, sent, ignore_period=True):
        if not isinstance(sent, str):
            raise TypeError(f"Input shout be a str but got {type(sent)} instead.")
        char_ids = []
        # This analyzer has the preprocessing routine built-in, which means that
        # the character pipeline will never see tokens that were replaced by
        # __NUM__ or __URL__.

        # The analyzer here might separate tokens, if passing tokens as sentences.
        # For example self.analyzer("C#") -> ['C', '#']
        for token in self.analyzer(sent):
            if token == "." and ignore_period:
                continue
            char_ids.append(self.token2char_ids(token))
        return char_ids

    def sents2char_ids(self, sents, ignore_period=True):
        if not isinstance(sents, list):
            raise TypeError(f"Expected list but got {type(sents)} instead.")

        sent_char_ids = []
        for sent in sents:
            sent_char_ids.append(
                self.sent2char_ids(sent, ignore_period=ignore_period)
            )
        return sent_char_ids
