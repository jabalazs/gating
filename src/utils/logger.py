import os

# import socket
import hashlib
import json
import subprocess

from datetime import datetime

# shouldn't do this!! the point is for this not to be library specific
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dataset as dt

from .io import write_output
from .gsheets import GsheetsClient
from .. import config


DATABASE_CONNECTION_STRING = config.DATABASE_CONNECTION_STRING
ARGS_TO_IGNORE_FOR_HASH_CALCULATION = {
    "datetime",
    "write_mode",
    "no_save_model",
    "log_interval",
    "spreadsheet",
}


def get_machine_id():
    try:
        with open("/etc/machine-id", "r") as f:
            # get machine-id and remove newline feed from the end
            machine_id = f.read()[:-1]
        return machine_id
    except IOError as e:
        print(
            "Machine_id file not found. Creating run hash without it."
            " {}".format(str(e))
        )


def get_server_name():
    try:
        with open("server_name", "r") as f:
            server_name = f.read().strip()
        return server_name
    except IOError as e:
        print("Server name not found. Creating run hash without it.")


def get_commit_hash():
    try:
        output = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
        return output.decode()
    except subprocess.CalledProcessError:
        print("Current project is not a git repo, omitting commit hash.")


class Logger(object):
    WRITE_MODES = ["FILE", "DATABASE", "BOTH", "NONE"]

    def __init__(
        self,
        args,
        model_name="",
        write_mode="BOTH",
        progress_bar=None,
        experiment_hash=None,
    ):
        """Class in charge of saving info about a model being trained
           args: argparse.Namespace """
        if write_mode not in self.WRITE_MODES:
            raise ValueError(
                "write_mode not recognized, try using one of the"
                "following: {}".format(self.WRITE_MODES)
            )
        self.run_day = datetime.now().strftime("%Y_%m_%d")
        self.run_time = datetime.now().strftime("%H_%M_%S")
        self.run_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.results_savepath = config.TRAINED_MODELS_PATH
        if not os.path.isdir(config.TRAINED_MODELS_PATH):
            os.makedirs(config.TRAINED_MODELS_PATH)

        self.write_mode = write_mode
        if self.write_mode == "NONE":
            self.write_mode = None

        self.args = vars(args)
        self.model_name = model_name
        self._complete_params(experiment_hash)
        self.run_savepath = os.path.join(self.results_savepath, self.args["hash"])

        if not os.path.isdir(self.run_savepath) and self.write_mode is not None:
            os.makedirs(self.run_savepath)

    def __getitem__(self, item):
        return self.args[item]

    def _complete_params(self, experiment_hash=None):
        """Add some more iformation to the args not provided through
           argparse"""

        self.args["datetime"] = self.run_datetime
        self.args["model_name"] = self.model_name

        self.args["server_name"] = get_server_name()
        self.args["commit"] = get_commit_hash()

        if not experiment_hash:
            # For hash based instead of datetime based monitoring
            hashargs = {
                arg: val
                for arg, val in self.args.items()
                if arg not in ARGS_TO_IGNORE_FOR_HASH_CALCULATION
            }

            hashobj = hashlib.sha1(
                json.dumps(hashargs, sort_keys=True).encode("utf8")
            )
            self.model_hash = hashobj.hexdigest()
        else:
            self.model_hash = experiment_hash

        self.args["hash"] = self.model_hash

    def write_hyperparams(self):
        write_mode = self.write_mode

        if write_mode == "FILE" or write_mode == "BOTH":
            hyperparam_file = os.path.join(self.run_savepath, "hyperparams.json")
            with open(hyperparam_file, "w") as f:
                f.write(json.dumps(self.args))

        if write_mode == "DATABASE" or write_mode == "BOTH":
            db = dt.connect(config.DATABASE_CONNECTION_STRING)
            runs_table = db["runs"]
            runs_table.upsert(self.args, keys="hash")

        if write_mode is None:
            pass

        if write_mode not in ("FILE", "DATABASE", "BOTH", None):
            raise ValueError(
                "{} mode not recognized. Try with FILE, DATABASE "
                "or BOTH".format(write_mode)
            )

    def _update_in_db(self, datadict):
        """expect a dictionary with the data to insert to the current run"""
        db = dt.connect(config.DATABASE_CONNECTION_STRING)
        runs_table = db["runs"]
        datadict["hash"] = self.model_hash
        runs_table.update(datadict, keys=["hash"])

    def update_results(self, datadict):
        """datadict: python dict"""
        if self.write_mode == "DATABASE" or self.write_mode == "BOTH":
            self._update_in_db(datadict)

    def read_from_database(self):
        db = dt.connect(config.DATABASE_CONNECTION_STRING)
        runs_table = db["runs"]
        datadict = runs_table.find_one(hash=self.model_hash)
        return datadict

    def insert_in_googlesheets(self):
        if self.write_mode == "DATABASE" or self.write_mode == "BOTH":
            print("Saving results in Google Spreadsheet")
            datadict = self.read_from_database()
            gsheets = GsheetsClient()
            gsheets.worksheet(config.SERVER_NAME).insert(datadict)
        else:
            print(
                f"Tried to save in Google Sheets but logger mode is not set "
                f"to DATABASE or BOTH."
            )

    def write_current_run_details(self, model=None):
        if not os.path.exists(config.LOG_PATH):
            os.makedirs(config.LOG_PATH)
        run_details_filename = os.path.join(config.LOG_PATH, "hyperparams.tmp")
        hyperparams = sorted(self.args.items())
        with open(run_details_filename, "w", encoding="utf-8") as f:
            for k, v in hyperparams:
                f.write("{}: {}\n".format(k, v))

        if model:
            architecture_filename = os.path.join(
                config.LOG_PATH, "architecture.tmp"
            )
            with open(architecture_filename, "w", encoding="utf8") as f:
                f.write(model)

    def torch_save_file(
        self, filename, obj, path_override=None, progress_bar=None
    ):
        """progress bar should be a tqdm instance with the `write` method"""
        if not os.path.isdir(self.run_savepath):
            os.makedirs(self.run_savepath)
        savepath = os.path.join(self.run_savepath, filename)
        savepath = path_override if path_override else savepath
        torch.save(obj, savepath)
        if progress_bar:
            progress_bar.write(f"File saved in {savepath}")
        return

    def save_image(self, numpy_matrix, filename="image", path_override=None):
        plt.imshow(numpy_matrix)
        savepath = os.path.join(self.run_savepath, filename + ".png")
        savepath = path_override if path_override else savepath
        plt.savefig(savepath)
        return

    def write_output_details(self, corpus, pred_label_ids, best_or_last, mode):
        idx2label = {idx: label for label, idx in corpus.label_dict.items()}
        idx2word = corpus.lang.index2word
        result = []
        for i, (id_tuple, pred_label_id) in enumerate(
            zip(corpus.id_tuples, pred_label_ids)
        ):
            prem_ids = id_tuple[0]
            hypo_ids = id_tuple[1]
            gold_label_id = id_tuple[2]
            pair_id = id_tuple[3]
            premise_words = map(idx2word.__getitem__, prem_ids)
            hypothesis_words = map(idx2word.__getitem__, hypo_ids)
            gold_label = idx2label[gold_label_id]
            pred_label = idx2label[pred_label_id]

            # Assumes the corpus is NOT shuffled
            genre = corpus.raw_examples[i].genre
            result.append(
                (
                    pair_id,
                    premise_words,
                    hypothesis_words,
                    gold_label,
                    pred_label,
                    genre,
                )
            )

        output = {
            "fields": [
                "pair_id",
                "premise",
                "hypothesis",
                "reference_label",
                "predicted_label",
                "genre",
            ],
            "result": result,
        }

        write_output(self.run_savepath, output, best_or_last, mode)
