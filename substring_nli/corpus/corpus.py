import os

from ..utils.io import load_or_create, read_jsonl, load_pickle
from .batch_iterator import BatchIterator

from .. import config


class BaseCorpus(object):
    def __init__(
        self,
        paths_dict,
        mode="train",
        use_chars=True,
        force_reload=False,
        train_data_proportion=1.0,
        valid_data_proportion=1.0,
        batch_size=64,
        shuffle_batches=False,
        batch_first=True,
    ):

        self.paths = paths_dict
        self.mode = mode

        self.use_chars = use_chars

        self.force_reload = force_reload

        self.train_data_proportion = train_data_proportion
        self.valid_data_proportion = valid_data_proportion

        self.batch_size = batch_size
        self.shuffle_batches = shuffle_batches
        self.batch_first = batch_first


class MultiNLICorpus(BaseCorpus):
    def __init__(
        self, *args, max_prem_length=None, max_hypo_length=None, **kwargs
    ):
        super(MultiNLICorpus, self).__init__(
            config.corpora_dict["multinli"], *args, **kwargs
        )

        try:
            token_dict = load_pickle(config.MULTINLI_TOKEN_DICT_PICKLE_PATH)
            char_dict = load_pickle(config.MULTINLI_CHAR_DICT_PICKLE_PATH)
            self.word2index = token_dict["token2id"]
            self.char2index = char_dict["char2id"]
        except FileNotFoundError:
            exit(
                "dict files not found. Try running the preprocessing "
                "script first"
            )

        basename = os.path.basename(self.paths["train"])
        filename_no_ext = os.path.splitext(basename)[0]
        train_pickle_path = os.path.join(
            config.CACHE_PATH, filename_no_ext + ".pkl"
        )
        self.train_data = load_or_create(
            train_pickle_path,
            read_jsonl,
            self.paths["train"],
            force_reload=self.force_reload,
        )

        # We use a set to eliminate duplicate entries
        self.label_ids = set([example["label_id"] for example in self.train_data])

        self.label_ids = list(self.label_ids)

        self.train_batches = BatchIterator(
            self.train_data,
            self.batch_size,
            data_proportion=self.train_data_proportion,
            shuffle=self.shuffle_batches,
            batch_first=self.batch_first,
            use_chars=self.use_chars,
        )

        basename = os.path.basename(self.paths["dev_matched"])
        filename_no_ext = os.path.splitext(basename)[0]
        dev_matched_pickle_path = os.path.join(
            config.CACHE_PATH, filename_no_ext + ".pkl"
        )
        self.dev_matched_data = load_or_create(
            dev_matched_pickle_path,
            read_jsonl,
            self.paths["dev_matched"],
            force_reload=self.force_reload,
        )

        self.dev_matched_batches = BatchIterator(
            self.dev_matched_data,
            self.batch_size,
            data_proportion=self.valid_data_proportion,
            shuffle=False,
            batch_first=self.batch_first,
            use_chars=self.use_chars,
        )

        # For compatibility
        self.dev_batches = self.dev_matched_batches

        basename = os.path.basename(self.paths["dev_mismatched"])
        filename_no_ext = os.path.splitext(basename)[0]
        dev_mismatched_pickle_path = os.path.join(
            config.CACHE_PATH, filename_no_ext + ".pkl"
        )
        self.dev_mismatched_data = load_or_create(
            dev_mismatched_pickle_path,
            read_jsonl,
            self.paths["dev_mismatched"],
            force_reload=self.force_reload,
        )

        self.dev_mismatched_batches = BatchIterator(
            self.dev_mismatched_data,
            self.batch_size,
            data_proportion=self.valid_data_proportion,
            shuffle=False,
            batch_first=self.batch_first,
            use_chars=self.use_chars,
        )


class SNLICorpus(BaseCorpus):
    def __init__(
        self, *args, max_prem_length=None, max_hypo_length=None, **kwargs
    ):
        super(SNLICorpus, self).__init__(
            config.corpora_dict["snli"], *args, **kwargs
        )

        try:
            token_dict = load_pickle(config.SNLI_TOKEN_DICT_PICKLE_PATH)
            char_dict = load_pickle(config.SNLI_CHAR_DICT_PICKLE_PATH)
            self.word2index = token_dict["token2id"]
            self.char2index = char_dict["char2id"]
        except FileNotFoundError:
            exit(
                "dict files not found. Try running the preprocessing "
                "script first"
            )

        basename = os.path.basename(self.paths["train"])
        filename_no_ext = os.path.splitext(basename)[0]
        train_pickle_path = os.path.join(
            config.CACHE_PATH, filename_no_ext + ".pkl"
        )
        self.train_data = load_or_create(
            train_pickle_path,
            read_jsonl,
            self.paths["train"],
            force_reload=self.force_reload,
        )

        # We use a set to eliminate duplicate entries
        self.label_ids = set([example["label_id"] for example in self.train_data])

        self.label_ids = list(self.label_ids)

        self.train_batches = BatchIterator(
            self.train_data,
            self.batch_size,
            data_proportion=self.train_data_proportion,
            shuffle=self.shuffle_batches,
            batch_first=self.batch_first,
            use_chars=self.use_chars,
        )

        basename = os.path.basename(self.paths["dev"])
        filename_no_ext = os.path.splitext(basename)[0]
        dev_pickle_path = os.path.join(
            config.CACHE_PATH, filename_no_ext + ".pkl"
        )
        self.dev_data = load_or_create(
            dev_pickle_path,
            read_jsonl,
            self.paths["dev"],
            force_reload=self.force_reload,
        )

        self.dev_batches = BatchIterator(
            self.dev_data,
            self.batch_size,
            data_proportion=self.valid_data_proportion,
            shuffle=False,
            batch_first=self.batch_first,
            use_chars=self.use_chars,
        )

        basename = os.path.basename(self.paths["test"])
        filename_no_ext = os.path.splitext(basename)[0]
        test_pickle_path = os.path.join(
            config.CACHE_PATH, filename_no_ext + ".pkl"
        )
        self.test_data = load_or_create(
            test_pickle_path,
            read_jsonl,
            self.paths["test"],
            force_reload=self.force_reload,
        )

        self.test_batches = BatchIterator(
            self.test_data,
            self.batch_size,
            data_proportion=1.0,
            shuffle=False,
            batch_first=self.batch_first,
            use_chars=self.use_chars,
        )
