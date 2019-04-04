import torch
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from .. import config
from ..seq_utils.pad import Padder
from ..utils.torch import to_var
from ..corpus.lang import Lang


class Tester(object):
    def __init__(self, model, hyperparams, batch_size=64, lang=None):
        """
        Provide functions for analyzing a pytorch model

        Parameters
        ----------
        model : Class inheriting from torch.nn.Module
            This model must provide and `encode` method with the
            following signature:
                args:
                    padded_word_ids : numpy.ndarray of shape (N, max_sent_len)
                    padded_char_ids : numpy.ndarray of shape (N, max_sent_len, max_word_len)
                    sent_lengths : torch.FloatTensor of shape (N)
                    word_lengths : torch.FloatTensor of shape (N, max_sent_len)
                kwargs:
                    masks : torch.FloatTensor of shape (N, max_sent_len)
        batch_size : int
        """
        self.model = model
        # This way we make sure to use the same lang with which the model was
        # trained
        self.lang = lang
        if self.lang is None:
            self.lang = Lang([], mode=hyperparams["corpus"], force_reload=False)

        self.batch_size = 64

        self.padder = Padder(config.PAD_ID)

    def sent_batch2input(self, batch, input_is_words=False):
        if input_is_words:
            word_ids = []
            for word in batch:
                try:
                    word_ids.append([self.lang.token2id[word]])
                except KeyError:
                    word_ids.append([config.UNK_ID])

        else:
            word_ids = self.lang.sents2ids(batch)

        char_ids = self.lang.sents2char_ids(batch)

        if input_is_words:
            # We flaten any representations that might have come from words
            # splitted by the lang: lang.sent2charids(["C#"]) -> ["C", "#"]
            new_ids = []
            for char_id in char_ids:
                # each char_id here is a list that should contain a single
                # element: a list of character ids
                if len(char_id) > 1:
                    char_id = [[item for sublist in char_id for item in sublist]]
                new_ids.append(char_id)

            char_ids = new_ids

        (padded_word_ids, sent_lengths, sent_masks) = self.padder.pad1d(word_ids)

        (padded_char_ids, _, word_lengths, word_masks) = self.padder.pad2d(
            char_ids
        )

        sent_lengths = to_var(
            torch.FloatTensor(sent_lengths), use_cuda=True, requires_grad=False
        )
        sent_masks = to_var(
            torch.FloatTensor(sent_masks), use_cuda=True, requires_grad=False
        )
        word_lengths = to_var(
            torch.FloatTensor(word_lengths), use_cuda=True, requires_grad=False
        )

        return (
            [padded_word_ids, padded_char_ids, sent_lengths, word_lengths],
            {"masks": sent_masks},
        )

    def sent_batch2vec(self, batch, input_is_words=False):
        args, kwargs = self.sent_batch2input(batch, input_is_words=input_is_words)

        # This way this function doesn't need to know anything about the input to the encode function.
        # We just need to define the function that transforms the batch into proper inputs for the specific model
        # we're using
        encoded = self.model.encode(*args, **kwargs)

        encoded = encoded.data.cpu().numpy()
        return encoded

    def sents2vec(self, sents, input_is_words=False):
        """
        Encode a list of sentences

        Parameters
        ----------
        sents: list
            A list of strings corresponding to sentences. Strings don't need to
            be tokenized because that's done in the Lang (although this
            shouldn't be the case)

        Returns
        -------
        A dict containing numpy.ndarray
            {'sent_repr': sent_vecs,
             'word_level_reprs': word_level_reprs,
             'char_level_reprs': char_level_reprs,
             'gates': gates
            }

        Notes
        -----
        Works with char_only, scalar_gate and vector_gate

        """
        sent_vecs = []
        gates = []
        word_level_reprs = []
        char_level_reprs = []
        # some sentences from samples are empty
        sents = [elem if elem != "" else "-" for elem in sents]
        for batch_idx in range(0, len(sents), self.batch_size):
            # print(f'Batch number {batch_idx}')

            curr_slice = sents[batch_idx : batch_idx + self.batch_size]

            encoded = self.sent_batch2vec(
                curr_slice, input_is_words=input_is_words
            )

            try:
                batch_gates = (
                    self.model.word_encoding_layer.word_encoding_layer.gate_result
                )
                batch_gates = batch_gates.data.cpu().numpy()
                gates.append(batch_gates)
            except AttributeError:
                # Models with no gates will raise an attribute error when calling
                # gate_result
                pass

            batch_word_level_reprs = (
                self.model.word_encoding_layer.word_encoding_layer.word_level_representations
            )
            batch_word_level_reprs = batch_word_level_reprs.data.cpu().numpy()
            word_level_reprs.append(batch_word_level_reprs)

            batch_char_level_reprs = (
                self.model.word_encoding_layer.word_encoding_layer.char_level_representations
            )
            if batch_char_level_reprs is not None:
                batch_char_level_reprs = batch_char_level_reprs.data.cpu().numpy()
                char_level_reprs.append(batch_char_level_reprs)

            sent_vecs.append(encoded)

        sent_vecs = np.vstack(sent_vecs)

        if len(gates) > 0:
            gates = np.concatenate(gates, axis=0)
        else:
            gates = None

        word_level_reprs = np.concatenate(word_level_reprs, axis=0)
        if len(char_level_reprs) > 0:
            char_level_reprs = np.concatenate(char_level_reprs, axis=0)

            # Maybe this assertion is not necessary anymore (raises error when
            # deliberately trying to work with smaller character representations)

            # assert word_level_reprs.shape == char_level_reprs.shape

        # vector gates should be of dim (batch_size, seq_len, hidden_dim)
        # scalar gates should be of dim (batch_size, seq_len, 1)

        return {
            "sent_repr": sent_vecs,
            "word_level_reprs": word_level_reprs,
            "char_level_reprs": char_level_reprs,
            "gates": gates,
        }

    def make_unk_words(self, sent):
        """Make unknown words more notorious by appending them a *

        Our models never see this modified token"""

        sent = sent.strip().split()
        new_sent = []
        for token in sent:
            if token not in self.lang.token2id.keys():
                token += "*"
            new_sent.append(token)
        return new_sent

    def get_insights(self, sents):
        """Receive a list of sentences and produce insights"""

        encoded_sents = self.sents2vec(sents)
        encoded = encoded_sents["sent_repr"]
        gates = encoded_sents["gates"]

        word_level_sents = [self.make_unk_words(sent) for sent in sents]

        ret_list = []
        for i, sent in enumerate(sents):
            gate_slice = None
            if not isinstance(gates, list) and gates is not None:
                gate_slice = gates[i, :, :]

            word_level_reprs = encoded_sents["word_level_reprs"][i, :, :]
            try:
                char_level_reprs = encoded_sents["char_level_reprs"][i, :, :]
            except TypeError:
                char_level_reprs = None

            ret_list.append(
                {
                    "raw_sent": sent,
                    "tokens": word_level_sents[i],
                    "sent_repr": encoded[i, :],
                    "gate": gate_slice,
                    "word_level_reprs": word_level_reprs,
                    "char_level_reprs": char_level_reprs,
                }
            )
        return ret_list

    def get_insights_from_words(self, words):
        """Receive a list of words and produce insights"""

        # We don't really care about the encoded sentences right now
        # but we want to update the gates by running a forward pass
        encoded_words = self.sents2vec(words, input_is_words=True)

        words = [self.make_unk_words(word) for word in words]
        words = [w for sublist in words for w in sublist]

        return {
            "raw_sent": " ".join(words),
            "tokens": words,
            "sent_repr": encoded_words["sent_repr"],
            "gate": encoded_words["gates"],
        }

    @staticmethod
    def plot_token_reprs(
        tokens,
        token_reprs,
        agg_repr=None,
        img_scaling=None,
        use_labels=True,
        vmin=0,
        vmax=1,
        ylabel=None,
    ):
        """
        Parameters
        ----------
        tokens : list of str
        token_reprs : numpy.ndarray of dim(seq_len, hidden_dim)
        agg_repr : (optional) numpy.ndarray of dim (seq_len, 1) """

        colormap = plt.cm.inferno  # or any other colormap
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        gate_norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        values = token_reprs[: len(tokens), :]

        if agg_repr is not None:
            agg_repr = agg_repr[: len(tokens), :]

        if img_scaling is None:
            img_scaling = 1

        fig = plt.figure()
        fig.set_size_inches(20, 2 * img_scaling)

        # Main figure
        left, bottom = 0, 0
        width, height = 1, 1
        ax = plt.axes([left, bottom, width, height])
        ax.set_yticks(np.arange(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=12)
        if not use_labels:
            ax.get_yaxis().set_visible(False)
        ax.set_xticks([])
        im = ax.imshow(values, norm=norm, cmap=colormap, aspect="auto")

        # Hidden dimension mean bar
        dim_bottom = bottom + height + 0.05 / img_scaling
        dim_height = 0.1 / img_scaling
        dim_ax = plt.axes([left, dim_bottom, width, dim_height])
        dim_ax.get_xaxis().set_visible(False)
        dim_ax.set_yticks(np.array([0]))
        dim_ax.set_yticklabels(["mean"], style="italic", fontsize=12)
        dim_ax.tick_params(axis="both", which="both", length=0)
        dim_ax.yaxis.tick_right()
        dim_mean_values = values.mean(axis=0, keepdims=True)
        dim_ax.imshow(dim_mean_values, norm=norm, cmap=colormap, aspect="auto")

        # Hidden dimension max bar
        max_dim_bottom = dim_bottom + dim_height + 0.05 / img_scaling
        max_dim_ax = plt.axes([left, max_dim_bottom, width, dim_height])
        max_dim_ax.get_xaxis().set_visible(False)
        max_dim_ax.set_yticks(np.array([0]))
        max_dim_ax.set_yticklabels(["max"], style="italic", fontsize=12)
        max_dim_ax.tick_params(axis="both", which="both", length=0)
        max_dim_ax.yaxis.tick_right()
        dim_max_values = values.max(axis=0, keepdims=True)
        max_dim_ax.imshow(dim_max_values, norm=norm, cmap=colormap, aspect="auto")
        #     max_dim_ax.imshow(values.max(axis=0, keepdims=True), norm=norm, cmap=colormap, aspect='equal')

        # sequence average bar
        seq_avg_bar_left = left + width + 0.005
        seq_avg_bar_width = 0.01
        seq_avg_bar_ax = plt.axes(
            [seq_avg_bar_left, bottom, seq_avg_bar_width, height]
        )
        seq_avg_bar_ax.get_yaxis().set_visible(False)
        seq_avg_bar_ax.set_xticks(np.array([0]))
        seq_avg_bar_ax.set_xticklabels(
            ["mean"], style="italic", fontsize=12, rotation=90
        )
        seq_avg_bar_ax.tick_params(axis="both", which="both", length=0)
        seq_mean_values = values.mean(axis=1, keepdims=True)
        seq_avg_bar_ax.imshow(
            seq_mean_values, norm=norm, cmap=colormap, aspect="auto"
        )

        # sequence max bar
        seq_max_bar_left = seq_avg_bar_left + seq_avg_bar_width + 0.005
        seq_max_bar_width = 0.01
        seq_max_bar_ax = plt.axes(
            [seq_max_bar_left, bottom, seq_max_bar_width, height]
        )
        seq_max_bar_ax.get_yaxis().set_visible(False)
        seq_max_bar_ax.set_xticks(np.array([0]))
        seq_max_bar_ax.set_xticklabels(
            ["max"], style="italic", fontsize=12, rotation=90
        )
        seq_max_bar_ax.tick_params(axis="both", which="both", length=0)
        seq_max_values = values.max(axis=1, keepdims=True)
        seq_max_bar_ax.imshow(
            seq_max_values, norm=norm, cmap=colormap, aspect="auto"
        )

        legend_left = seq_max_bar_left + 0.03
        legend_height = height + 2 * (dim_height + (0.05 + 0.01) / img_scaling)
        legend_width = 0.01
        legend_ax = plt.axes([legend_left, bottom, legend_width, legend_height])
        fig.colorbar(im, cax=legend_ax)

        if agg_repr is not None:
            # external aggregate repr bar
            ax.get_yaxis().set_visible(False)
            ext_repr_bar_width = 0.01
            ext_repr_bar_left = left - ext_repr_bar_width - 0.005
            ext_repr_bar_ax = plt.axes(
                [ext_repr_bar_left, bottom, ext_repr_bar_width, height]
            )
            ext_repr_bar_ax.set_yticks(np.arange(len(tokens)))
            ext_repr_bar_ax.set_yticklabels(tokens, fontsize=12)
            if ylabel:
                ext_repr_bar_ax.set_ylabel(ylabel, fontsize=20)
            if not use_labels:
                ext_repr_bar_ax.get_yaxis().set_visible(False)
            ext_repr_bar_ax.set_xticks(np.array([0]))
            ext_repr_bar_ax.set_xticklabels(
                ["scalar_gate"], style="italic", fontsize=12, rotation=90
            )
            ext_repr_bar_ax.tick_params(axis="both", which="both", length=0)
            ext_repr_bar_ax.imshow(
                agg_repr, norm=gate_norm, cmap=colormap, aspect="auto"
            )

        return {
            "values": values,
            "dim_mean_values": dim_mean_values,
            "dim_max_values": dim_max_values,
            "seq_mean_values": seq_mean_values,
            "seq_max_values": seq_max_values,
            "tokens": tokens,
            "main_ax": ax,
            "figure": fig,
        }

    def plot_sent(
        self,
        sent,
        scalar_tester=None,
        img_scaling=1.0,
        use_labels=True,
        char_vmin=0,
        char_vmax=1,
        autoscale_color=False,
    ):
        insights = self.get_insights([sent])
        scalar_insights_gate = None

        if scalar_tester is not None:
            scalar_insights = scalar_tester.get_insights([sent])
            scalar_insights_gate = scalar_insights[0]["gate"]

        tokens = insights[0]["tokens"]
        word_lvl_reprs = insights[0]["word_level_reprs"]
        char_lvl_reprs = insights[0]["char_level_reprs"]
        vector_gate = insights[0]["gate"]

        ret_dict = {}

        vmin = 0
        vmax = 1
        if autoscale_color:
            vmin = min(word_lvl_reprs.min(), char_lvl_reprs.min())
            vmax = max(word_lvl_reprs.max(), char_lvl_reprs.max())
            char_vmin = vmin
            char_vmax = vmax

        ret_dict["word_level"] = self.plot_token_reprs(
            tokens,
            word_lvl_reprs,
            scalar_insights_gate,
            img_scaling=img_scaling,
            use_labels=use_labels,
            vmin=vmin,
            vmax=vmax,
            ylabel="Word Level",
        )

        if char_lvl_reprs is not None:
            # vmin=-1e-4, vmax=2.5e-1
            ret_dict["char_level"] = self.plot_token_reprs(
                tokens,
                char_lvl_reprs,
                scalar_insights_gate,
                img_scaling=img_scaling,
                use_labels=use_labels,
                vmin=char_vmin,
                vmax=char_vmax,
                ylabel="Character Level",
            )

        if vector_gate is not None:
            ret_dict["vector_gate"] = self.plot_token_reprs(
                tokens,
                vector_gate,
                scalar_insights_gate,
                img_scaling=img_scaling,
                use_labels=use_labels,
                vmin=0,
                vmax=1,
                ylabel="Gate",
            )

            gate_result = ((1 - vector_gate) * word_lvl_reprs) + (
                vector_gate * char_lvl_reprs
            )
            ret_dict["gate_result"] = self.plot_token_reprs(
                tokens,
                gate_result,
                scalar_insights_gate,
                img_scaling=img_scaling,
                use_labels=use_labels,
                vmin=0,
                vmax=1,
                ylabel="Gate Result",
            )

        return ret_dict

    def plot_sents(self, sents, **kwargs):
        ret_dicts = []
        for sent in sents:
            ret_dict = self.plot_sent(sent, **kwargs)
            ret_dicts.append(ret_dict)
        return ret_dicts
