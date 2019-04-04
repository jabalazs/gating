import numpy as np
import os

from .. import config
from ..utils.io import load_or_create, get_name_from_path

np.random.seed(4242)


class Embeddings(object):
    """Minimal abstraction for pre-trained embeddings.
    This class maps token -> vector"""

    def __init__(self, filepath, k_most_frequent=None, force_reload=False):
        """filepath: path of the plain text file containing the pre-trained
        embeddings"""
        self.filepath = filepath
        embeddings_name = get_name_from_path(filepath)
        embeddings_pickle_filepath = os.path.join(
            config.CACHE_PATH, embeddings_name + ".pkl"
        )

        self.dict_repr = load_or_create(
            embeddings_pickle_filepath,
            self._format_raw_data,
            k_most_frequent,
            force_reload=force_reload,
        )

        self.tokens = set(self.dict_repr.keys())
        self.emb_vocab_size = len(self.tokens)
        # Some embedding files come with a header (fasttext) which is why we
        # explore row 1 instead of 0
        self.embedding_dim = len(list(self.dict_repr.values())[1])
        self.unknown_tokens = []
        # self.embedding_matrix = None

    def _format_raw_data(self, k_most_frequent=None):
        dict_repr = {}
        with open(self.filepath, "rb") as f:
            i = 0
            for line in f:
                linelist = str(line.rstrip(), "utf8", errors="strict").split(" ")
                token = linelist[0]
                vector = linelist[1:]
                vector = np.array(vector, dtype=np.float32)
                dict_repr[token] = vector
                i += 1
                if k_most_frequent and i == k_most_frequent:
                    # This assumes that vectors in the embedding file are
                    # sorted according to their frquency (such as GloVe)
                    break
        return dict_repr

    def generate_embedding_matrix(self, word2index):
        """Generate a numpy ndarray of dim (vocabulary_size, embedding_dim),
        where vocabulary size correspond to the amount of elements in word2index
        and embedding_dim is the size of the vectors in the pre-trained
        embeddings.

        The row index will correspond to the index given by word2index
        NOTE: Do not confuse emb_vocab_size with vocab size. Usually the
        embedding matrix to be used in runtime will have a subset of the tokens
        in the pre-trained embeddings
        """
        vocab_size = len(word2index)
        embedding_matrix = np.zeros((vocab_size, self.embedding_dim))
        unknown_tokens = []

        for token, index in word2index.items():

            if token in config.SPECIAL_TOKENS:
                embedding_matrix[index] = np.random.uniform(
                    -0.05, 0.05, size=[self.embedding_dim]
                )
                continue

            try:
                embedding_matrix[index] = self.dict_repr[token]
            except KeyError:
                embedding_matrix[index] = np.random.uniform(
                    -0.05, 0.05, size=[self.embedding_dim]
                )
                unknown_tokens.append(token)
            except IndexError:
                import ipdb

                ipdb.set_trace(context=10)

        self.unknown_tokens = unknown_tokens
        # self.embedding_matrix = embedding_matrix
        return embedding_matrix

    def __getitem__(self, key):
        return self.dict_repr[key]
