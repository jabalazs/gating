import torch
import colored_traceback

import numpy as np

from tqdm import tqdm

from substring_nli.corpus.corpus import MultiNLICorpus, SNLICorpus
from substring_nli.corpus.embeddings import Embeddings
from substring_nli.utils.logger import Logger
from substring_nli.utils.torch import normalize_embeddings
from substring_nli.trainer import Trainer
from substring_nli.optim.optim import OptimWithDecay
from substring_nli import config
from substring_nli.layers.layers import CharEncoder

from substring_nli.models.base import (
    NLIClassifier,
    WordEncodingLayer,
    WordCharEncodingLayer,
    SentenceEncodingLayer,
)
from substring_nli.models.helpers import evaluate

from base_args import base_parser, CustomArgumentParser

colored_traceback.add_hook(always=True)


base_parser.description = "PyTorch MultiNLI Inner Attention Classifier"
arg_parser = CustomArgumentParser(
    parents=[base_parser], description="PyTorch MultiNLI"
)

arg_parser.add_argument(
    "--model",
    type=str,
    default="infersent",
    choices=SentenceEncodingLayer.SENTENCE_ENCODING_METHODS,
    help="Model to use",
)

arg_parser.add_argument(
    "--corpus",
    type=str,
    default="snli",
    choices=list(config.corpora_dict.keys()),
    help="Name of the corpus to use.",
)

arg_parser.add_argument(
    "--mismatched_dev",
    action="store_true",
    help="Whether to use the mismatched dev dataset for " "MultiNLI",
)

arg_parser.add_argument(
    "--embeddings",
    type=str,
    default="glove",
    choices=list(config.embedding_dict.keys()) + ["random"],
    help="Name of the embeddings to use.",
)

arg_parser.add_argument(
    "--finetune_word_embeddings",
    "-fwe",
    default=1,
    type=int,
    help="Whether to train word embeddings or not. "
    "1: train word embeddings, 0: don't train them.",
)

arg_parser.add_argument(
    "--embeddings_norm_dim",
    "-end",
    default=None,
    type=int,
    help="Which dimension to use for L2 normalizing "
    "word and char embeddings. Options are None (no normalization), "
    "0 (normalization accross embedding dimensions),"
    "1 (normalization accross vocabulary)",
)

arg_parser.add_argument(
    "--lstm_hidden_size",
    type=int,
    default=2048,
    help="Hidden dimension size for the word-level LSTM",
)

arg_parser.add_argument(
    "--char_hidden_size",
    "-chs",
    default=None,
    type=int,
    help="Hidden dimension size for the character-level LSTM",
)

arg_parser.add_argument(
    "--force_reload",
    action="store_true",
    help="Whether to reload pickles or not (makes the "
    "process slower, but ensures data coherence)",
)
arg_parser.add_argument(
    "--char_emb_dim", type=int, default=50, help="Char embedding dimension"
)
arg_parser.add_argument(
    "--pooling_method",
    type=str,
    default="max",
    choices=["mean", "sum", "last", "max"],
    help="Pooling scheme to use as raw sentence " "representation method.",
)
arg_parser.add_argument(
    "--dropout",
    type=float,
    default=0.1,
    help="Dropout applied to layers. 0 means no dropout.",
)

arg_parser.add_argument(
    "--char_fw_bw_agg_method",
    "-cfbam",
    default="linear_sum",
    type=str,
    choices=CharEncoder.FORWARD_BACKWARD_AGGREGATION_METHODS,
    help="How to combine the forward and backward lstm "
    "passes when encoding words as a sequence of characters",
)

arg_parser.add_argument(
    "--word_encoding_method",
    "-wem",
    type=str,
    default="embed",
    choices=WordEncodingLayer.WORD_ENCODING_METHODS,
    help="How to obtain word representations",
)

arg_parser.add_argument(
    "--word_char_aggregation_method",
    "-wcam",
    choices=WordCharEncodingLayer.AGGREGATION_METHODS,
    default=None,
    help="Way in which character-level and word-level word "
    "representations are aggregated",
)

arg_parser.add_argument(
    "--spreadsheet",
    "-ss",
    action="store_true",
    help="Save results in google spreadsheet",
)


def validate_args(hp):
    """hp: argparser parsed arguments. type: Namespace"""

    if (
        hp.word_encoding_method == "char_lstm"
        and not hp.word_char_aggregation_method
    ):
        raise ValueError(
            f"Need to pass a word_char_aggregation_method when "
            f"using char_lstm word_encoding_method. "
            f"Choose one from {WordCharEncodingLayer.AGGREGATION_METHODS}"
        )

    if hp.mismatched_dev and hp.corpus != "multinli":
        raise ValueError(
            "mismatched_dev flag passed but a corpus other than "
            "multinli is being used. Either disable "
            "mismatched_dev or choose multinli as corpus"
        )

    model_is_gating = (
        hp.word_encoding_method == "char_lstm"
        and hp.word_char_aggregation_method in ["scalar_gate", "vector_gate"]
    )

    if model_is_gating and hp.char_hidden_size is not None:
        raise Exception(
            f"You are using a {hp.word_char_aggregation_method} "
            "gating mechanism, but you "
            "passed a specific character hidden size. "
            "Either change the word_encoding_method or do "
            "not declare a character hidden size"
        )

    if not model_is_gating and hp.char_hidden_size is None:
        # Using 300 because of historical reasons: This has always been the
        # hardcoded default because we needed this dim to be compatible with
        # GloVe pre-trained embeddings
        hp.char_hidden_size = 150 if hp.char_fw_bw_agg_method == "cat" else 300
        print(f"Using default char_hidden_size={hp.char_hidden_size}")

        # raise Exception('Please specify a character hidden size with the '
        #                 '--char_hidden_size or -chs flags.')

    if hp.finetune_word_embeddings not in [0, 1]:
        raise Exception("--finetune_word_embeddings can only be 0 or 1")

    if hp.finetune_word_embeddings == 0 and hp.embeddings == "random":
        raise Exception(
            "Are you sure you want to initialize embeddings "
            "randomly and not train them?"
        )


def main():
    hp = arg_parser.parse_args()
    validate_args(hp)

    logger = Logger(hp, model_name="Baseline", write_mode=hp.write_mode)
    print(f"Running experiment {logger.model_hash}.")
    if hp.write_mode != "NONE":
        logger.write_hyperparams()
        print(
            f"Hyperparameters and checkpoints will be saved in "
            f"{logger.run_savepath}"
        )

    torch.manual_seed(hp.seed)
    torch.cuda.manual_seed_all(hp.seed)  # silently ignored if there are no GPUs

    CUDA = False
    if torch.cuda.is_available() and not hp.no_cuda:
        CUDA = True

    SAVE_IN_SPREADSHEET = False
    if hp.spreadsheet:
        SAVE_IN_SPREADSHEET = True

    SAVE_MODEL = False
    if hp.write_mode != "NONE" and not hp.no_save_model:
        SAVE_MODEL = True

    # these must match the ones found in config.corpora_dict
    if hp.corpus == "snli":
        CorpusClass = SNLICorpus
    elif hp.corpus == "multinli":
        CorpusClass = MultiNLICorpus

    corpus = CorpusClass(
        force_reload=hp.force_reload,
        train_data_proportion=hp.train_data_proportion,
        valid_data_proportion=hp.dev_data_proportion,
        batch_size=hp.batch_size,
    )

    if hp.embeddings != "random":
        # Load pre-trained embeddings
        embeddings = Embeddings(
            config.embedding_dict[hp.embeddings],
            k_most_frequent=None,
            force_reload=hp.force_reload,
        )

        # Get subset of embeddings corresponding to our vocabulary
        embedding_matrix = embeddings.generate_embedding_matrix(corpus.word2index)
        print(
            f"{len(embeddings.unknown_tokens)} words from vocabulary not found "
            f"in {hp.embeddings} embeddings. "
        )
    else:
        word_vocab_size = len(corpus.word2index)
        embedding_matrix = np.random.uniform(
            -0.05, 0.05, size=(word_vocab_size, 300)
        )

    # Initialize torch Embedding object with subset of pre-trained embeddings
    torch_embeddings = torch.nn.Embedding(*embedding_matrix.shape)
    torch_embeddings.weight = torch.nn.Parameter(torch.Tensor(embedding_matrix))
    if hp.finetune_word_embeddings == 0:
        torch_embeddings.weight.requires_grad = False

    if hp.embeddings_norm_dim and hp.embeddings != "random":
        # Only normalize if word embeddings are pre-trained
        torch_embeddings = normalize_embeddings(torch_embeddings, dim=1, order=2)

    # Repeat process for character embeddings with the difference that they are
    # not pretrained

    # Initialize character embedding matrix randomly
    char_vocab_size = len(corpus.char2index)

    # FIXME: We should set numpy's random seed in this script, however we
    # haven't done so because it's been set in the embeddings file since the
    # beginning. We'll keep it this way for compatibility
    char_embedding_matrix = np.random.uniform(
        -0.05, 0.05, size=(char_vocab_size, hp.char_emb_dim)
    )
    char_torch_embeddings = torch.nn.Embedding(*char_embedding_matrix.shape)
    char_torch_embeddings.weight = torch.nn.Parameter(
        torch.Tensor(char_embedding_matrix)
    )

    num_classes = len(corpus.label_ids)
    batch_size = corpus.train_batches.batch_size
    model = NLIClassifier(
        num_classes,
        batch_size,
        torch_embeddings=torch_embeddings,
        char_embeddings=char_torch_embeddings,
        word_encoding_method=hp.word_encoding_method,
        char_fw_bw_agg_method=hp.char_fw_bw_agg_method,
        word_char_aggregation_method=hp.word_char_aggregation_method,
        sent_encoding_method=hp.model,
        hidden_sizes=hp.lstm_hidden_size,
        char_hidden_size=hp.char_hidden_size,
        use_cuda=CUDA,
        pooling_method=hp.pooling_method,
        batch_first=True,
        dropout=hp.dropout,
    )

    if CUDA:
        model.cuda()
    logger.write_current_run_details(str(model))

    optimizer = OptimWithDecay(
        model.parameters(),
        method=hp.optim,
        initial_lr=hp.learning_rate,
        max_grad_norm=hp.grad_clipping,
    )

    loss_function = torch.nn.CrossEntropyLoss()

    trainer = Trainer(
        model,
        optimizer,
        loss_function,
        num_epochs=hp.epochs,
        use_cuda=CUDA,
        log_interval=hp.log_interval,
    )

    # Whether to use matched or mismatched MultiNLI data
    dev_batches = corpus.dev_batches
    if hp.mismatched_dev:
        dev_batches = corpus.dev_mismatched_batches

    try:
        best_accuracy = None
        for epoch in tqdm(range(hp.epochs), desc="Epoch"):
            # Train a single epoch
            trainer.train_epoch(
                corpus.train_batches,
                epoch,
                embeddings_norm_dim=hp.embeddings_norm_dim,
            )

            # Evaluate
            tqdm.write("Evaluating...")
            eval_dict = evaluate(model, dev_batches)

            # Update learning rate
            optim_updated, new_lr = trainer.optimizer.updt_lr_accuracy(
                epoch, eval_dict["accuracy"]
            )
            if new_lr < 1e-5:
                break
            if optim_updated:
                tqdm.write(f"Learning rate decayed to {new_lr}")

            # Store model and accuracy, depending on accuracy
            accuracy = eval_dict["accuracy"]
            if not best_accuracy or accuracy > best_accuracy:
                best_accuracy = accuracy
                logger.update_results(
                    {"best_valid_acc": best_accuracy, "best_epoch": epoch}
                )
                if SAVE_MODEL:
                    # See
                    # https://pytorch.org/docs/stable/notes/serialization.html
                    # It should be enough to save the state dict, but that would
                    # require refactoring the test.py and
                    # scripts/evaluate_word_sim.py scripts, that rely on the
                    # entire model having been saved.
                    logger.torch_save_file(
                        "best_model_state_dict.pth",
                        model.state_dict(),
                        progress_bar=tqdm,
                    )
                    logger.torch_save_file(
                        "best_model.pth", model, progress_bar=tqdm
                    )
    except KeyboardInterrupt:
        pass
    finally:
        if SAVE_IN_SPREADSHEET:
            print("Saving in google spreadsheet")
            logger.insert_in_googlesheets()


if __name__ == "__main__":
    main()
