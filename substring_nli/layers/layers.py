import torch
import torch.nn as nn

from ..utils.torch import pack_forward
from .pooling import GatherLastLayer


class CharEncoder(nn.Module):
    FORWARD_BACKWARD_AGGREGATION_METHODS = ["cat", "linear_sum"]

    def __init__(
        self,
        char_embedding_dim,
        hidden_size,
        char_fw_bw_agg_method="cat",
        bidirectional=True,
        train_char_embeddings=True,
        use_cuda=True,
    ):

        if char_fw_bw_agg_method not in self.FORWARD_BACKWARD_AGGREGATION_METHODS:
            raise ValueError(
                f"{char_fw_bw_agg_method} not recognized, try with one of "
                f"{self.FORWARD_BACKWARD_AGGREGATION_METHODS}"
            )

        super(CharEncoder, self).__init__()
        self.char_embedding_dim = char_embedding_dim
        self.n_layers = 1
        self.char_hidden_dim = hidden_size
        self.bidirectional = bidirectional
        self.num_dirs = 2 if bidirectional else 1
        self.hidden_x_dirs = self.num_dirs * self.char_hidden_dim
        self.use_cuda = use_cuda
        self.char_lstm = nn.LSTM(
            self.char_embedding_dim,
            self.char_hidden_dim,
            self.n_layers,
            bidirectional=self.bidirectional,
            dropout=0.0,
        )

        self.gather_last = GatherLastLayer(
            self.char_hidden_dim, bidirectional=self.bidirectional
        )

        self.char_fw_bw_agg_method = char_fw_bw_agg_method

        if self.char_fw_bw_agg_method == "cat":
            self.out_dim = self.hidden_x_dirs

        elif self.char_fw_bw_agg_method == "linear_sum":
            self.out_dim = self.char_hidden_dim
            self.linear_layer = nn.Linear(
                self.hidden_x_dirs, self.char_hidden_dim
            )

    def forward(self, char_batch, word_lengths):
        """char_batch: (batch_size, seq_len, word_len, char_emb_dim)
           word_lengths: (batch_size, seq_len)"""

        (batch_size, seq_len, word_len, char_emb_dim) = char_batch.size()

        # (batch_size * seq_len, word_len, char_emb_dim)
        char_batch = char_batch.view(batch_size * seq_len, word_len, char_emb_dim)

        # (batch_size, seq_len) -> (batch_size * seq_len)
        word_lengths = word_lengths.view(batch_size * seq_len)

        # (batch_size * seq_len, word_len, hidden_x_dirs)
        word_lvl_repr = pack_forward(self.char_lstm, char_batch, word_lengths)

        # (batch_size * seq_len, hidden_x_dirs)
        word_lvl_repr = self.gather_last(word_lvl_repr, lengths=word_lengths)

        # last dimension of gather_last will always correspond to concatenated
        # last hidden states of lstm if bidirectional

        # (batch_size, seq_len, hidden_x_dirs)
        word_lvl_repr = word_lvl_repr.view(
            batch_size, seq_len, self.hidden_x_dirs
        )

        # We store this tensor for future introspection
        self.concat_fw_bw_reprs = word_lvl_repr.clone()

        if self.char_fw_bw_agg_method == "linear_sum":
            # Based on the paper: http://www.anthology.aclweb.org/D/D16/D16-1209.pdf
            # Line below is W*word_lvl_repr + b which is equivalent to
            # [W_f; W_b] * [h_f;h_b] + b which in turn is equivalent to
            # W_f * h_f + W_b * h_b + b
            word_lvl_repr = self.linear_layer(word_lvl_repr)

        return word_lvl_repr


class LinearAggregationLayer(nn.Module):
    def __init__(self, in_dim):
        """

        Simply concatenate the provided tensors on their last dimension
        which needs to have the same size,  along with their
        element-wise multiplication and difference

        Taken from the paper:
             "Learning Natural Language Inference using Bidirectional
             LSTM model and Inner-Attention"
             https://arxiv.org/abs/1605.09090

        """
        super(LinearAggregationLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = 4 * in_dim

    def forward(self, input_1, input_2):
        """

        :param : input_1
            Size is (*, hidden_size)

        :param input_2:
            Size is (*, hidden_size)

        :return:

            Merged vectors, size is (*, 4*hidden size)
        """
        assert input_1.size(-1) == input_2.size(-1)
        mult_combined_vec = torch.mul(input_1, input_2)
        diff_combined_vec = torch.abs(input_1 - input_2)
        # cosine_sim = simple_columnwise_cosine_similarity(input_1, input_2)
        # cosine_sim = cosine_sim.unsqueeze(1)
        # euclidean_dist = distance(input_1, input_2, 2)

        combined_vec = torch.cat(
            (input_1, input_2, mult_combined_vec, diff_combined_vec),
            input_1.dim() - 1,
        )

        return combined_vec
