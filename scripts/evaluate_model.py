import os
import sys
import argparse
import json

from pathlib import Path

project_root = Path(__file__).absolute().parent.parent
sys.path.append(str(project_root))

import torch

import numpy as np

from src import config
from src.corpus.lang import Lang
from src.seq_utils.pad import Padder
from src.utils.torch import to_var

import colored_traceback

colored_traceback.add_hook(always=True)

PATH_SENTEVAL = "SentEval/"
PATH_TRANSFER_TASKS = "SentEval/data/senteval_data/"

parser = argparse.ArgumentParser(description="NLI training")
parser.add_argument("modelpath", type=str, help="Path to model")
parser.add_argument(
    "-gpu", "--gpu_index", type=int, default=0, help="Which gpu to use"
)
params, _ = parser.parse_known_args()

# import senteval
sys.path.insert(0, PATH_SENTEVAL)
import senteval

params, _ = parser.parse_known_args()

# free_gpu_index = get_free_gpu_index(max_memory=1000)
# torch.cuda.device(params.gpu_index)


class dotdict(dict):
    """ dot.notation access to dictionary attributes """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Tester(object):
    padder = Padder(config.PAD_ID)

    def __init__(self, model, lang):

        self.is_cuda = True
        self.lang = lang
        self.model = model

    def encode(self, samples, batch_size=64, tokenize=False):

        embeddings = []
        # some sentences from samples are empty
        samples = [elem if elem != "" else "-" for elem in samples]
        for batch_idx in range(0, len(samples), batch_size):

            curr_slice = samples[batch_idx : batch_idx + batch_size]

            word_ids = self.lang.sents2ids(curr_slice)
            char_ids = self.lang.sents2char_ids(curr_slice)

            (padded_word_ids, sent_lengths, sent_masks) = self.padder.pad1d(
                word_ids
            )

            try:
                (
                    padded_char_ids,
                    _,
                    word_lengths,
                    word_masks,
                ) = self.padder.pad2d(char_ids)
            except ValueError:
                import ipdb

                ipdb.set_trace(context=10)

            sent_lengths = to_var(
                torch.FloatTensor(sent_lengths),
                use_cuda=True,
                requires_grad=False,
            )
            sent_masks = to_var(
                torch.FloatTensor(sent_masks), use_cuda=True, requires_grad=False
            )
            word_lengths = to_var(
                torch.FloatTensor(word_lengths),
                use_cuda=True,
                requires_grad=False,
            )

            encoded = self.model.encode(
                padded_word_ids,
                padded_char_ids,
                sent_lengths,
                word_lengths,
                masks=sent_masks,
            )

            encoded = encoded.data.cpu().numpy()

            embeddings.append(encoded)

        embeddings = np.vstack(embeddings)
        return embeddings


def batcher(params, batch):
    # batch contains list of words
    sentences = [" ".join(s) for s in batch]

    # this encode method is the one defined in this file
    embeddings = params.infersent.encode(sentences, batch_size=64, tokenize=False)
    return embeddings


"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

# define transfer tasks
transfer_tasks = [
    "MR",
    "CR",
    "SUBJ",
    "MPQA",
    "SST2",
    "SST5",
    "TREC",
    "SICKRelatedness",
    "SICKEntailment",
    "STS16",
    "STSBenchmark",
]

# define senteval params
params_senteval = dotdict(
    {
        "usepytorch": True,
        "task_path": PATH_TRANSFER_TASKS,
        "seed": 1111,
        "kfold": 5,
    }
)


def get_corpus_name_from_modelpath(modelpath):
    dirpath, basename = os.path.split(modelpath)
    hyperparam_filepath = os.path.join(dirpath, "hyperparams.json")
    try:
        with open(hyperparam_filepath, "r") as f:
            hyperparams = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Are you you sure you provided the full path to the trained "
            "model?\nIt should look like\n"
            "data/trained_models/<model_hash>/best_model.pth\nYou provided\n"
            f"{modelpath}"
        )

    if hyperparams["corpus"] not in ["snli", "multinli"]:
        raise Exception(
            "Detected corpus is neither snli nor multinli, are you "
            "sure you wanted to run this script?"
        )

    return hyperparams["corpus"]


def main():

    modelpath = params.modelpath

    model_hash = os.path.split(os.path.dirname(modelpath))[1]

    print(f"Evaluating {modelpath}")

    corpus_name = get_corpus_name_from_modelpath(modelpath)
    print(f"Model trained with the {corpus_name} dataset")
    lang = Lang([], mode=corpus_name, min_freq_threshold=0, force_reload=False)

    try:
        model = torch.load(modelpath)
        tester = Tester(model, lang)

        params_senteval.infersent = tester

        se = senteval.engine.SE(params_senteval, batcher)
        results_transfer = se.eval(transfer_tasks)

    except AttributeError as e:
        # AttributeError will be raised when trying to evaluate an old model
        # that doesn't have an essential property
        print(f"Could not evaluate {modelpath}. Got the following error: \n{e}")
        exit()
    except KeyboardInterrupt:
        exit()

    model_dir = os.path.split(modelpath)[0]
    results_path = os.path.join(model_dir, "senteval_exp_results.json")
    with open(results_path, "w") as f:
        json.dump(results_transfer, f)
        print(f"Results saved in {results_path}")

    print("Done!")


if __name__ == "__main__":
    main()
