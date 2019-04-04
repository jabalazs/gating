import torch

from torch import nn

from ..layers.pooling import PoolingLayer
from ..layers.layers import LinearAggregationLayer, CharEncoder

from ..utils.torch import to_var, pack_forward


class CharEmbeddingLayer(nn.Module):
    def __init__(self, embeddings, use_cuda=True):
        super(CharEmbeddingLayer, self).__init__()
        self.embeddings = embeddings
        self.use_cuda = use_cuda
        self.embedding_dim = embeddings.embedding_dim

    def forward(self, np_batch):
        """np_batch: (batch_size, seq_len, word_len)"""
        batch = to_var(
            torch.LongTensor(np_batch),
            use_cuda=self.use_cuda,
            requires_grad=False,
        )

        batch_size, seq_len, word_len = batch.size()
        batch = batch.view(batch_size, seq_len * word_len)

        emb_batch = self.embeddings(batch)

        emb_batch = emb_batch.view(
            batch_size, seq_len, word_len, self.embedding_dim
        )
        return emb_batch


class WordEmbeddingLayer(nn.Module):
    def __init__(self, embeddings, use_cuda=True):
        super(WordEmbeddingLayer, self).__init__()
        self.embeddings = embeddings
        self.use_cuda = use_cuda
        self.embedding_dim = embeddings.embedding_dim

    def forward(self, np_batch, char_batch=None, word_lengths=None):
        """np_batch: (batch_size, seq_len)"""
        batch = to_var(
            torch.LongTensor(np_batch),
            use_cuda=self.use_cuda,
            requires_grad=False,
        )

        emb_batch = self.embeddings(batch)

        # We save representations for later inspection
        self.word_level_representations = emb_batch
        self.char_level_representations = None

        return emb_batch


class WordCharEncodingLayer(nn.Module):
    AGGREGATION_METHODS = ["char_only", "cat", "scalar_gate", "vector_gate"]

    def __init__(
        self,
        word_embeddings,
        char_embeddings,
        char_hidden_size=50,
        char_fw_bw_agg_method="linear_sum",
        word_char_aggregation_method="cat",
        train_char_embeddings=True,
        use_cuda=True,
    ):

        aggregation_method = word_char_aggregation_method
        if aggregation_method not in self.AGGREGATION_METHODS:
            raise RuntimeError(
                f"{aggregation_method} method not recogized. "
                f"Try one of {self.AGGREGATION_METHODS})"
            )

        super(WordCharEncodingLayer, self).__init__()
        self.word_embeddings = word_embeddings
        self.char_embeddings = char_embeddings
        self.char_hidden_size = char_hidden_size
        self.char_fw_bw_agg_method = char_fw_bw_agg_method
        self.aggregation_method = aggregation_method
        self.train_char_embeddings = train_char_embeddings
        self.use_cuda = use_cuda

        self.word_embedding_layer = WordEmbeddingLayer(
            word_embeddings, use_cuda=self.use_cuda
        )
        self.char_embedding_layer = CharEmbeddingLayer(
            char_embeddings, use_cuda=self.use_cuda
        )

        self.char_encoding_layer = CharEncoder(
            char_embeddings.embedding_dim,
            self.char_hidden_size,
            char_fw_bw_agg_method=self.char_fw_bw_agg_method,
            bidirectional=True,
            train_char_embeddings=True,
            use_cuda=self.use_cuda,
        )

        if self.aggregation_method == "char_only":
            self.embedding_dim = self.char_encoding_layer.out_dim

        elif self.aggregation_method == "cat":
            # we add these dimensions because we are going to concatenate the vector reprs
            self.embedding_dim = (
                self.char_encoding_layer.out_dim + word_embeddings.embedding_dim
            )

        elif self.aggregation_method == "scalar_gate":
            self.embedding_dim = self.char_encoding_layer.out_dim
            self.scalar_gate = nn.Linear(self.char_encoding_layer.out_dim, 1)

        elif self.aggregation_method == "vector_gate":
            self.embedding_dim = self.char_encoding_layer.out_dim
            self.vector_gate = nn.Linear(
                self.char_encoding_layer.out_dim, self.char_encoding_layer.out_dim
            )

    def forward(self, word_batch, char_batch, word_lengths):
        emb_word_batch = self.word_embedding_layer(word_batch)
        emb_char_batch = self.char_embedding_layer(char_batch)

        char_lvl_word_repr = self.char_encoding_layer(
            emb_char_batch, word_lengths
        )

        # We are storing these values in self so we can later access them in the
        # trained models
        self.word_level_representations = emb_word_batch
        self.char_level_representations = char_lvl_word_repr

        if self.aggregation_method == "char_only":
            word_reprs = char_lvl_word_repr
            self.gate_result = None

        elif self.aggregation_method == "cat":
            word_reprs = torch.cat([emb_word_batch, char_lvl_word_repr], 2)
            self.gate_result = None

        # FIXME: Both gating mechanisms below assume that emb_word_batch and
        # char_lvl_word_repr have the same hidden dimensions

        # We are also storing self.gate_result to be able to visualize them
        # once the model is trained

        elif self.aggregation_method == "scalar_gate":
            gate_result = torch.sigmoid(
                self.scalar_gate(emb_word_batch)
            )  # in [0; 1]
            word_reprs = (
                1.0 - gate_result
            ) * emb_word_batch + gate_result * char_lvl_word_repr
            self.gate_result = gate_result

        elif self.aggregation_method == "vector_gate":
            gate_result = torch.sigmoid(
                self.vector_gate(emb_word_batch)
            )  # in [0; 1]
            word_reprs = (
                1.0 - gate_result
            ) * emb_word_batch + gate_result * char_lvl_word_repr
            self.gate_result = gate_result

        return word_reprs


class WordEncodingLayer(nn.Module):

    WORD_ENCODING_METHODS = ["embed", "char_lstm"]

    @staticmethod
    def factory(word_encoding_method, *args, **kwargs):
        if word_encoding_method == "embed":
            kwargs.pop("char_embeddings")
            kwargs.pop("char_hidden_size")
            kwargs.pop("train_char_embeddings")
            kwargs.pop("word_char_aggregation_method")
            kwargs.pop("char_fw_bw_agg_method")
            return WordEmbeddingLayer(*args, **kwargs)
        if word_encoding_method == "char_lstm":
            return WordCharEncodingLayer(*args, **kwargs)

    def __init__(self, word_encoding_method, *args, **kwargs):
        super(WordEncodingLayer, self).__init__()
        self.word_encoding_method = word_encoding_method
        if self.word_encoding_method not in self.WORD_ENCODING_METHODS:
            raise AttributeError(
                f"`{self.word_encoding_method}` not "
                f"recognized. Try using "
                f"one of {self.WORD_ENCODING_METHODS}"
            )

        self.word_encoding_layer = self.factory(
            self.word_encoding_method, *args, **kwargs
        )

        self.embedding_dim = self.word_encoding_layer.embedding_dim

    def __call__(self, *args, **kwargs):
        return self.word_encoding_layer(*args, **kwargs)


class BLSTMEncoder(nn.Module):
    """
    Args:
        embedding_dim: """

    def __init__(
        self,
        embedding_dim,
        hidden_sizes=2048,
        num_layers=1,
        bidirectional=True,
        dropout=0.0,
        batch_first=True,
        use_cuda=True,
    ):
        super(BLSTMEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_sizes  # sizes in plural for compatibility
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_dirs = 2 if bidirectional else 1
        self.dropout = dropout
        self.batch_first = batch_first
        self.out_dim = self.hidden_size * self.num_dirs

        self.enc_lstm = nn.LSTM(
            self.embedding_dim,
            self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True,
            dropout=self.dropout,
        )

    def is_cuda(self):
        # either all weights are on cpu or they are on gpu
        return "cuda" in str(type(self.enc_lstm.bias_hh_l0))

    def forward(self, emb_batch, lengths):
        """Based on: https://github.com/facebookresearch/InferSent/blob/4b7f9ec7192fc0eed02bc890a56612efc1fb1147/models.py

           Take an embedded batch of dim (batch_size, seq_len, embedding_dim) and pass
           it through the RNN. Return a tensor of dim (batch_size, seq_len, out_dim)
           where out dim depends on the hidden dim of the RNN and its directions"""

        sent_output = pack_forward(self.enc_lstm, emb_batch, lengths)

        return sent_output


class SentenceEncodingLayer(nn.Module):

    SENTENCE_ENCODING_METHODS = ["infersent"]

    @staticmethod
    def factory(sent_encoding_method, *args, **kwargs):
        if sent_encoding_method == "infersent":
            return BLSTMEncoder(*args, **kwargs)

    def __init__(self, sent_encoding_method, *args, **kwargs):
        super(SentenceEncodingLayer, self).__init__()
        self.sent_encoding_method = sent_encoding_method

        if self.sent_encoding_method not in self.SENTENCE_ENCODING_METHODS:
            raise AttributeError(
                f"`{self.sent_encoding_method}` not "
                f"recognized. Try using "
                f"one of {self.SENTENCE_ENCODING_METHODS}"
            )

        self.sent_encoding_layer = self.factory(
            self.sent_encoding_method, *args, **kwargs
        )
        self.out_dim = self.sent_encoding_layer.out_dim

    def __call__(self, *args, **kwargs):
        return self.sent_encoding_layer(*args, **kwargs)


class NLIClassifier(nn.Module):
    """Args:
        embeddings: torch word embeddings
        """

    def __init__(
        self,
        num_classes,
        batch_size,
        torch_embeddings=None,
        char_embeddings=None,
        word_encoding_method="embed",
        char_fw_bw_agg_method="linear_sum",
        word_char_aggregation_method=None,
        sent_encoding_method="infersent",
        hidden_sizes=None,
        char_hidden_size=None,
        pooling_method="max",
        batch_first=True,
        dropout=0.0,
        use_cuda=True,
    ):

        super(NLIClassifier, self).__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.embeddings = torch_embeddings
        self.batch_first = batch_first
        self.dropout = dropout

        self.use_cuda = use_cuda

        self.pooling_method = pooling_method

        self.word_encoding_method = word_encoding_method
        self.word_char_aggregation_method = word_char_aggregation_method
        self.char_fw_bw_agg_method = char_fw_bw_agg_method
        self.sent_encoding_method = sent_encoding_method
        self.hidden_sizes = hidden_sizes
        self.char_hidden_size = char_hidden_size

        self.char_embeddings = None
        if char_embeddings:
            self.char_embeddings = char_embeddings

        model_is_gating = (
            self.word_encoding_method == "char_lstm"
            and self.word_char_aggregation_method
            in ["scalar_gate", "vector_gate"]
        )

        if model_is_gating:
            # FIXME: Hack for having compatible dimensions. The line below
            # assumes there are only 2 possible char_fw_bw_method: cat and
            # linear_sum. 300 corresponds to the glove embeddings dimensions
            # characters are going to be combined with. We define 150 when
            # using cat so it becomes 300 when concatenating forward and
            # backward passes
            self.char_hidden_size = 150 if char_fw_bw_agg_method == "cat" else 300

        self.word_encoding_layer = WordEncodingLayer(
            self.word_encoding_method,
            self.embeddings,
            char_embeddings=self.char_embeddings,
            char_hidden_size=self.char_hidden_size,
            char_fw_bw_agg_method=self.char_fw_bw_agg_method,
            word_char_aggregation_method=self.word_char_aggregation_method,
            train_char_embeddings=True,
            use_cuda=self.use_cuda,
        )

        self.sent_encoding_layer = SentenceEncodingLayer(
            self.sent_encoding_method,
            self.word_encoding_layer.embedding_dim,
            hidden_sizes=self.hidden_sizes,
            batch_first=self.batch_first,
            use_cuda=self.use_cuda,
        )

        sent_encoding_dim = self.sent_encoding_layer.out_dim
        self.pooling_layer = PoolingLayer(self.pooling_method, sent_encoding_dim)

        self.sent_aggregation_layer = LinearAggregationLayer(
            self.pooling_layer.out_dim
        )

        self.dense_layer = nn.Sequential(
            nn.Linear(self.sent_aggregation_layer.out_dim, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, self.num_classes),
        )

    def encode(
        self,
        batch,
        char_batch,
        sent_lengths,
        word_lengths,
        masks=None,
        embed_words=True,
    ):
        """ Encode a batch of ids into a sentence representation.

            This method exists for compatibility with facebook's senteval

            batch: padded batch of word indices if embed_words, else padded
                  batch of torch tensors corresponding to embedded word
                  vectors"""
        if embed_words:
            embedded = self.word_encoding_layer(batch, char_batch, word_lengths)
        else:
            embedded = batch
        sent_embedding = self.sent_encoding_layer(embedded, sent_lengths)
        agg_sent_embedding = self.pooling_layer(
            sent_embedding, lengths=sent_lengths, masks=masks
        )
        return agg_sent_embedding

    def batch2input(self, batch, sent_type):
        """Transform a batch object into an input for the model

        Parameters
        ----------
        batch : A batch object returned by BatchIterator

        sent_type : str, {prem, hypo}
            whether to prepare premises or hypotheses (SNLI specific because of
            the way the expected data from BatchIterator is structured.)

        Returns
        -------
        dict containing logits
        """
        if sent_type not in ["prem", "hypo"]:
            raise Exception(
                f"sent_type: {sent_type} not recognized. Try using "
                f"`prem` or `hypo`."
            )

        sents = batch[f"{sent_type}s"]
        sent_lengths = batch[f"{sent_type}_sent_lengths"]
        masks = batch[f"{sent_type}_masks"]

        sent_lengths = to_var(
            torch.LongTensor(sent_lengths), self.use_cuda, requires_grad=False
        )
        masks = to_var(
            torch.FloatTensor(masks), self.use_cuda, requires_grad=False
        )

        char_sequences = None
        word_lengths = None
        if self.char_embeddings:
            char_sequences = batch[f"{sent_type}_char_sequences"]
            word_lengths = batch[f"{sent_type}_word_lengths"]
            char_masks = batch[f"{sent_type}_char_masks"]

            word_lengths = to_var(
                torch.LongTensor(word_lengths), self.use_cuda, requires_grad=False
            )

        return (
            [sents],
            {
                "char_batch": char_sequences,
                "sent_lengths": sent_lengths,
                "word_lengths": word_lengths,
                "masks": masks,
            },
        )

    def forward(self, batch):

        prem_args, prem_kwargs = self.batch2input(batch, "prem")
        hypo_args, hypo_kwargs = self.batch2input(batch, "hypo")

        pooled_prem = self.encode(*prem_args, **prem_kwargs)
        pooled_hypo = self.encode(*hypo_args, **hypo_kwargs)

        pair_repr = self.sent_aggregation_layer(pooled_prem, pooled_hypo)
        logits = self.dense_layer(pair_repr)

        ret_dict = {"logits": logits}

        return ret_dict
