import os
import json
import argparse

from glob import glob

import numpy as np
import torch

from src import config
from src.corpus.corpus import SNLICorpus
from src.models.helpers import evaluate
from src.utils.ops import np_softmax

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument(
    "model_hash",
    type=str,
    help="Hash of the model to test, can be a partial hash",
)

arg_parser.add_argument(
    "--batch_size",
    "-bs",
    default=64,
    type=int,
    help="Batch size. Note: Batch size here will not affect final results.",
)

arg_parser.add_argument(
    "--save_sent_reprs",
    "-ssr",
    action="store_true",
    default=False,
    help="Save sentence representations in the experiment directory.",
)


def main():

    hp = arg_parser.parse_args()

    experiment_path = os.path.join(
        config.TRAINED_MODELS_PATH, hp.model_hash + "*"
    )
    ext_experiment_path = glob(experiment_path)
    assert len(ext_experiment_path) == 1, "Try provinding a longer model hash"
    ext_experiment_path = ext_experiment_path[0]
    model_path = os.path.join(ext_experiment_path, "best_model.pth")
    model = torch.load(model_path)

    model_hyperparams_path = os.path.join(ext_experiment_path, "hyperparams.json")
    with open(model_hyperparams_path, "r") as f:
        hyperparams = json.load(f)

    # these must match the ones found in config.corpora_dict
    if hyperparams["corpus"] == "snli":
        CorpusClass = SNLICorpus
    elif hyperparams["corpus"] == "multinli":
        raise NotImplementedError("MultiNLI dataset does not have test set")

    corpus = CorpusClass(force_reload=False, batch_size=hp.batch_size)

    print(f"Testing model {model_path}")

    eval_dict = evaluate(model, corpus.test_batches)

    probs = np_softmax(eval_dict["output"])
    probs_filepath = os.path.join(ext_experiment_path, "test_probabilities.csv")
    np.savetxt(probs_filepath, probs, delimiter=",", fmt="%.8f")
    print(f"Saved prediction probs in {probs_filepath}")

    labels_filepath = os.path.join(ext_experiment_path, "test_predictions.txt")
    labels = [label + "\n" for label in eval_dict["labels"]]
    with open(labels_filepath, "w", encoding="utf-8") as f:
        f.writelines(labels)
    print(f"Saved prediction file in {labels_filepath}")

    if hp.save_sent_reprs:
        representations_filepath = os.path.join(
            ext_experiment_path, "sentence_representations.txt"
        )

        with open(representations_filepath, "w", encoding="utf-8") as f:
            np.savetxt(
                representations_filepath,
                eval_dict["sent_reprs"],
                delimiter=" ",
                fmt="%.8f",
            )


if __name__ == "__main__":
    main()
