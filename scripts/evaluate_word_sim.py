import os
import sys
import random
import json
import argparse

from pathlib import Path

project_root = Path(__file__).absolute().parent.parent
sys.path.append(str(project_root))

import pandas as pd

import torch

from substring_nli.utils.tester import Tester
from substring_nli.corpus.lang import Lang
from substring_nli.utils.io import load_pickle
from substring_nli.models.base import NLIClassifier
from substring_nli.config import CACHE_PATH

from substring_nli.utils.word_similarity import evaluate_similarity_in_all

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_dirs",
    "-d",
    default=None,
    type=str,
    required=True,
    nargs="+",
    help="Directories containing all the experiment directories",
)
parser.add_argument(
    "--target_file",
    "-t",
    default="word_evaluation_results.csv",
    type=str,
    help="Path of the results file, including its name",
)

parser.add_argument(
    "--force",
    "-f",
    action="store_true",
    default=False,
    help="Whether to overwrite results file",
)
args = parser.parse_args()


def create_model(hyperparams, word_dict, char_dict):

    num_words = len(word_dict["token2id"])
    num_chars = len(char_dict["char2id"])

    # Create model with current code
    word_embeddings = torch.nn.Embedding(num_words, 300)
    char_embeddings = torch.nn.Embedding(num_chars, 50)

    model = NLIClassifier(
        num_classes=3,
        batch_size=64,
        torch_embeddings=word_embeddings,
        char_embeddings=char_embeddings,
        word_encoding_method=hyperparams["word_encoding_method"],
        char_fw_bw_agg_method="linear_sum",
        word_char_aggregation_method=hyperparams["word_char_aggregation_method"],
        sent_encoding_method="infersent",
        hidden_sizes=2048,
        char_hidden_size=300,
        use_cuda=True,
        pooling_method="max",
        batch_first=True,
        dropout=hyperparams["dropout"],
    )

    model.cuda()
    return model


def test_old_model(model_dir, dicts_dir):

    """Function for getting a tester for an old model

    I needed to create this function to load a model with
    current code and load old weights from a state_dict.
    I also needed to provide an Lang instance with an old vocab

    This function expects pickles called <dataset>_train_<type>_dict.pkl where
    dataset = {mnli, snli} and type = {token, char}, in the dicts_dir,
    corresponding to dicts containing word (char) counts, their mappings to ids,
    and the tokenizer used when training. This data is later fed to the lang to
    conver new inputs into ids compatible with the models.

    """

    print(f"testing model in {model_dir}")

    # Load hyperparameters
    hyperparams_path = os.path.join(model_dir, "hyperparams.json")
    with open(hyperparams_path) as infile:
        hyperparams = json.load(infile)

    if hyperparams["corpus"] == "multinli":
        dataset = "mnli"
    elif hyperparams["corpus"] == "snli":
        dataset = "snli"

    token_dict_pickle_path = os.path.join(
        dicts_dir, f"{dataset}_train_token_dict.pkl"
    )
    char_dict_pickle_path = os.path.join(
        dicts_dir, f"{dataset}_train_char_dict.pkl"
    )

    word_dict = load_pickle(token_dict_pickle_path)
    char_dict = load_pickle(char_dict_pickle_path)

    try:
        wcam = hyperparams["word_char_aggregation_method"]
    except KeyError:
        print(
            "word_char_aggregation_method not found in hyperparams. "
            "Assuming None"
        )
        wcam = None
        hyperparams["word_char_aggregation_method"] = wcam

    model = create_model(hyperparams, word_dict, char_dict)

    state_dict = model.state_dict()

    # Get old state dict and replace current model's weights with checkpoint's
    old_state_dict_path = os.path.join(model_dir, "best_model_state_dict.pth")
    old_state_dict = torch.load(old_state_dict_path)

    state_dict.update(old_state_dict)

    model.load_state_dict(state_dict)

    lang = Lang(
        [],
        token_dict_pickle_path=token_dict_pickle_path,
        char_dict_pickle_path=char_dict_pickle_path,
        force_reload=False,
    )

    tester = Tester(model, hyperparams, lang=lang)

    ret_list = evaluate_similarity_in_all(tester, mode=wcam)

    for dict_ in ret_list:
        dict_["method"] = wcam if wcam is not None else "word_only"
        dict_["corpus"] = dataset
        dict_["hash"] = hyperparams["hash"]

    return ret_list


def main():
    results = []

    # args.model_dirs contain a list of directories. Each of these directories
    # contain subdirectories corresponding to experiment runs
    for models_dir in args.model_dirs:
        dir_elems = os.listdir(models_dir)

        abs_dir_elems = [os.path.join(models_dir, elem) for elem in dir_elems]
        dirs = [dir_elem for dir_elem in abs_dir_elems if os.path.isdir(dir_elem)]

        for dir_ in dirs:
            results += test_old_model(dir_, CACHE_PATH)

    results_df = pd.DataFrame(results)

    print(f"Saving results in {args.target_file}")
    results_df.to_csv(args.target_file, index=False)


if __name__ == "__main__":

    if os.path.exists(args.target_file) and not args.force:
        print(
            f"File {args.target_file} already exists. To overwrite it, run this "
            f"script with the --force (-f) flag."
        )
        exit()

    main()
