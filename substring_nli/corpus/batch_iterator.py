import random

from .. import config
from ..seq_utils.pad import Padder

random.seed(1111)


class BaseNLPBatch(dict):
    def __init__(self, *args, **kwargs):
        super(BaseNLPBatch, self).__init__(*args, **kwargs)

        self.batch_first = kwargs.pop("batch_first")
        self.batch_size = kwargs.pop("batch_size")
        self.padder = Padder(config.PAD_ID)

    def _pad1d(self, sequences, *args, **kwargs):
        """sequences: a list of lists"""
        padded_sequences, lengths, masks = self.padder.pad1d(
            sequences, *args, **kwargs
        )
        if not self.batch_first:
            padded_sequences = padded_sequences.transpose(1, 0)
            masks = masks.transpose(1, 0)

        return padded_sequences, lengths, masks

    def _pad2d(self, sequences2d, *args, **kwargs):
        """sequences2d: a list of lists of lists"""
        (
            padded_sequences2d,
            sent_lengths,
            word_lengths,
            char_masks,
        ) = self.padder.pad2d(sequences2d, *args, **kwargs)
        if not self.batch_first:
            padded_sequences2d = padded_sequences2d.transpose(1, 2, 0)
            char_masks = char_masks.transpose(1, 2, 0)

        return padded_sequences2d, sent_lengths, word_lengths, char_masks


class MultiNLIBatch(BaseNLPBatch):
    def __init__(self, examples, *args, **kwargs):
        super(MultiNLIBatch, self).__init__(*args, **kwargs)
        self.examples = examples
        self.use_chars = kwargs.pop("use_chars")
        self._build_batch_from_examples()

    def _build_batch_from_examples(self):

        # This class expects examples to be a list containing dicts
        # with at least a 'sequence', a 'labels' key and a 'char_sequence'
        # if use_chars is true
        ids = [example["pairID"] for example in self.examples]

        prem_sequences = [example["prem_token_ids"] for example in self.examples]
        prem_padded_sequences, prem_lengths, prem_masks = self._pad1d(
            prem_sequences
        )

        hypo_sequences = [example["hypo_token_ids"] for example in self.examples]
        hypo_padded_sequences, hypo_lengths, hypo_masks = self._pad1d(
            hypo_sequences
        )

        self["prems"] = prem_padded_sequences
        self["prem_sent_lengths"] = prem_lengths
        self["prem_masks"] = prem_masks

        self["hypos"] = hypo_padded_sequences
        self["hypo_sent_lengths"] = hypo_lengths
        self["hypo_masks"] = hypo_masks

        self["ids"] = ids

        try:
            # We try this because test examples have no labels
            labels = [example["label_id"] for example in self.examples]
            self["labels"] = labels
        except KeyError:
            pass

        if self.use_chars:
            prem_char_sequences = [
                example["prem_char_ids"] for example in self.examples
            ]
            hypo_char_sequences = [
                example["hypo_char_ids"] for example in self.examples
            ]

            (
                prem_padded_sequences2d,
                prem_sent_lengths,
                prem_word_lengths,
                prem_char_masks,
            ) = self._pad2d(prem_char_sequences)

            (
                hypo_padded_sequences2d,
                hypo_sent_lengths,
                hypo_word_lengths,
                hypo_char_masks,
            ) = self._pad2d(hypo_char_sequences)

            self["prem_char_sequences"] = prem_padded_sequences2d
            self["prem_word_lengths"] = prem_word_lengths
            self["prem_char_masks"] = prem_char_masks

            self["hypo_char_sequences"] = hypo_padded_sequences2d
            self["hypo_word_lengths"] = hypo_word_lengths
            self["hypo_char_masks"] = hypo_char_masks

    def inspect(self):
        for key, value in self.items():
            try:
                print(f"{key}: shape={value.shape}")
            except AttributeError:
                if isinstance(value, list):
                    print(f"{key}: length={len(value)}")
                elif isinstance(value, bool) or isinstance(value, int):
                    print(f"{key}: {value}")
                else:
                    print(f"{key}: type={type(value)}")

    def __repr__(self):
        return self.__class__.__name__


class BatchIterator(object):
    def __init__(
        self,
        examples,
        batch_size,
        data_proportion=1.0,
        shuffle=False,
        batch_first=False,
        use_chars=False,
    ):

        """Create batches of length batch_size from the examples

        Parameters
        ----------
        examples : iterable
            The data to be batched. Independent from corpus or model
        batch_size : int
            The desired batch size.
        data_proportion :
        shuffle : bool
            whether to shuffle the data before creating the batches.
        batch_first : bool
        use_chars : bool
        """

        self.examples = examples
        self.batch_size = batch_size

        self.data_proportion = data_proportion
        self.shuffle = shuffle
        self.batch_first = batch_first

        self.use_chars = use_chars

        if shuffle:
            random.shuffle(self.examples)

        self.examples_subset = self.examples

        assert 0.0 < data_proportion <= 1.0
        self.n_examples_to_use = int(
            len(self.examples_subset) * self.data_proportion
        )

        self.examples_subset = self.examples_subset[: self.n_examples_to_use]

        self.num_batches = (self.n_examples_to_use + batch_size - 1) // batch_size

        self.labels = []
        self.ids = []

        self.num_batches = (
            len(self.examples_subset) + batch_size - 1
        ) // batch_size

    def __getitem__(self, index):
        assert index < self.num_batches, (
            "Index is greater "
            "than the number of batches "
            "%d>%d" % (index, self.num_batches)
        )

        # First we obtain the batch slices
        examples_slice = self.examples_subset[
            index * self.batch_size : (index + 1) * self.batch_size
        ]

        return MultiNLIBatch(
            examples_slice,
            batch_size=self.batch_size,
            batch_first=self.batch_first,
            use_chars=self.use_chars,
        )

    def __len__(self):
        return self.num_batches
