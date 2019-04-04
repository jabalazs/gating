import random

import torch
import numpy as np

from scipy.stats import pearsonr, spearmanr

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


from ..third_party.web.datasets.similarity import (
    fetch_WS353,
    fetch_MEN,
    fetch_SimLex999,
    fetch_MTurk,
    fetch_MTurk771,
    fetch_RG65,
    fetch_RW,
    fetch_VerbSim3500,
    fetch_Card660,
)


def encode_words(words, tester, mode):
    """
    words: a list of words
    tester: tester implemented in my `tester.py`"""

    encoded_words = tester.sents2vec(words, input_is_words=True)

    #     final_repr = encoded_words['sent_repr']
    words = encoded_words["word_level_reprs"].squeeze(1)
    chars = encoded_words["char_level_reprs"]

    AGGREGATION_METHODS = ["char_only", "cat", "scalar_gate", "vector_gate", None]
    if mode not in AGGREGATION_METHODS:
        raise Exception(f"mode {mode} not recognized")

    if mode is None:
        final_repr = words

    if mode == "char_only":
        chars = chars.squeeze(1)
        final_repr = chars

    if mode == "cat":
        chars = chars.squeeze(1)
        final_repr = np.concatenate((words, chars), axis=1)

    if mode == "scalar_gate" or mode == "vector_gate":
        chars = chars.squeeze(1)
        gate = encoded_words["gates"].squeeze(1)
        final_repr = ((1 - gate) * words) + (gate * chars)

    return {
        "word_reprs": words,
        # "char_reprs": chars,
        #         "gates": gate,
        "final_reprs": final_repr,
    }


def encode_dataset(dataset, tester, mode="gate"):
    """
    dataset: object from the `word-embeddings-benchmarks` repo
        dataset.X: a list of lists of pairs of word
        dataset.y: similarity between these pairs

    tester: tester implemented in my `tester.py`"""

    words_1 = [x[0] for x in dataset["X"]]

    encoded_words_1 = encode_words(
        words_1, tester, mode=mode
    )
    encoded_words_2 = encode_words(
        [x[1] for x in dataset["X"]], tester, mode=mode
    )

    return encoded_words_1, encoded_words_2


def cosine_similarity(A, B):
    """
    Calculate the cosine similarity between the rows of A and B
    A: numpy.ndarray of shape (elems, dim)
    B: numpy.ndarray of shape (elems, dim)"""
    scores = np.array(
        [
            v1.dot(v2.T) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            for v1, v2 in zip(A, B)
        ]
    )
    return scores


def evaluate_similarity(dataset, tester, mode="gate"):
    A, B = encode_dataset(dataset, tester, mode=mode)
    repr_levels = A.keys()

    results = {}
    for repr_lvl in repr_levels:
        if repr_lvl == "gates":
            continue
        sim = cosine_similarity(A[repr_lvl], B[repr_lvl])
        pearson_corr = pearsonr(sim, dataset["y"])
        spearman_corr = spearmanr(sim, dataset["y"])
        results[repr_lvl] = {
            "pearson": pearson_corr[0],
            "spearman": spearman_corr.correlation,
        }
    return results


def evaluate_similarity_in_all(tester, mode="gate"):
    similarity_tasks = {
        "MEN": fetch_MEN(),
        "WS353": fetch_WS353(),
        "WS353R": fetch_WS353(which="relatedness"),
        "WS353S": fetch_WS353(which="similarity"),
        "SimLex999": fetch_SimLex999(),
        "RW": fetch_RW(),
        "RG65": fetch_RG65(),
        "MTurk287": fetch_MTurk(),
        "MTurk771": fetch_MTurk771(),
        "VerbSim3500": fetch_VerbSim3500(),
        "Card660": fetch_Card660(),
        #         "TR9856": fetch_TR9856(),
    }

    # MEN y values come in a different format, so we fix it here
    similarity_tasks["MEN"].y = similarity_tasks["MEN"].y.squeeze(1)

    results = []
    for task, dataset in similarity_tasks.items():
        dataset_results = evaluate_similarity(dataset, tester, mode=mode)
        pearson_delta = (
            dataset_results["final_reprs"]["pearson"]
            - dataset_results["word_reprs"]["pearson"]
        )
        spearman_delta = (
            dataset_results["final_reprs"]["spearman"]
            - dataset_results["word_reprs"]["spearman"]
        )
        result_dict = {
            "dataset": task,
            "pearson": dataset_results["final_reprs"]["pearson"] * 100,
            "spearman": dataset_results["final_reprs"]["spearman"] * 100,
            "pearson_delta": pearson_delta * 100,
            "spearman_delta": spearman_delta * 100,
        }
        results.append(result_dict)

    return results
