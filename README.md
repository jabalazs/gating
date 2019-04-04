
# Gating Mechanisms

Code accompanying the paper "Gating Mechanisms for Combining Character and
Word-level Word Representations: An Empirical Study". If you use use this code
please consider citing the paper.


## Installation instructions

1. Clone this repo                                                                                                           

   ```bash
   git clone https://github.com/jabalazs/substring_nli
   cd gating
   ```

2. Create a conda environment and activate it

   ```bash
   conda create -n <name> python=3.6
   conda activate <name>
   ```

   Where you can replace `<name>` by whatever you want.

2. Setup everything
   ```bash
   scripts/setup.sh
   ```
   This script will do the following:
   * Install all the dependencies to the currently-active conda environment
   * Get the SNLI and MultiNLI datasets and store them in `data/corpora`
   * Download the `glove.840B.300d.txt` pretrained word embeddings and store them
     in `data/word_embeddings`.
   * Preprocess the SNLI and MultiNLI datasets, and create preprocessed files in
     `data/preprocessed`, and some cache files in `.cache`.
   * Finally, it will execute SentEval's script for getting and
     preprocessing its own data.

   Downloading and preprocessing everything will take a while (~35 min),
   depending on your internet connection and your conda cache.

3. Run the training script

   ```bash
   python train.py
   ```

   * The first time this is run, some more cache files will be created in `.cache`
   * To test the whole pipeline quickly you can train on a small subset of SNLI
     by running the previous command with the `--train_data_proportion=0.01`
     (`-tdp=0.01`) flag 

To see a list of available commands and hyperparameters execute

```bash
python train.py --help
```

## Saving Trained Models

By default, each time a model with a new set of hyperparameters is trained the
code will:
* Calculate a hash `<hash>` from them
* Create the directory `data/trained_models/<hash>`
* Create the file `data/trained_models/<hash>/hyperparams.json` containing the
hyperparameters.
* Add an entry to the sqlite3 database `data/trained_models/runs.db` (and create
  it if it doesn't exist), containing the same information as `hyperparams.json`
* Create the `log/hyperparams.tmp` and `log/architecture.tmp` with information
  about the model being currently trained (or last trained).

After the end of each epoch, the code will:
* Validate the trained model
* Create the checkpoints `data/trained_models/<hash>/best_model.pth` and
  `data/trained_models/<hash>/best_model_state_dict.pth` if the current accuracy
  is better than in the previous epoch, or if it is the first epoch.
* Update the sqlite database with validation information

We also provide a script for visualizing relevant entries in the database:
`data/trained_models/lsruns`. Execute it from the same directory where `runs.db`
is located, after at least one evaluation step.

At the moment it is not possible to resume training properly. Only
model information is stored, not the optimizer state.

## Gating Mechanisms

For training different gating mechanisms modify the `--word_encoding_method`
(`-wem`) and `--word_char_aggregation_method` (`-wcam`) flags when executing
`train.py`

* `word_encoding_method` defines whether to use word embeddings alone (`embed`),
  or to use a combination of these with character representations (`char_lstm`)
* When using `--word_encoding_method=char_lstm`, `word_char_aggregation_method`
  defines how to combine the word embeddings with the character-level word
  embeddings. Possible options are:
  - `char_only`: Ignore word embeddings; use only word representations coming
    from characters
  - `cat`: concatenate word and character representations
  - `scalar_gate`: use a scalar gate for combining them
  - `vector_gate`: use a vector gate for combining them

Please refer to the paper or to the code for more details.

## SentEval - Evaluating Sentence Representations

To evaluate a trained model using `SentEval` execute:

```bash
python scripts/evaluate_model.py data/trained_models/<hash>/best_model.pth
```

This will generate the file `data/trained_models/<hash>/senteval_exp_results.json`
containing the evaluation results.

## Word Embeddings Benchmarks

To evaluate all trained models using `word-embeddings-benchmarks` execute:

```bash
python scripts/evaluate_word_sim.py --models_dir data/trained_models
```

The first time this is executed, the `~/web_data` directory will be created.

After evaluating every model in `data/trained_models`, the file
`word_evaluation_results.csv` will be created.



## Citation
```
@InProceedings{balazs2019gating,
  author       = {Balazs, Jorge A. and 
                  Matsuo, Yutaka},
  title        = {{Gating Mechanisms for Combining Character and
                   Word-level Word Representations: An Empirical Study}},
  booktitle    = {Proceedings of the 2019 Conference of the North American
                  Chapter of the Association for Computational Linguistics:
                  Student Research Workshop},
  year         = {2019},
  address      = {Minneapolis, Minnesota, USA},
  month        = {June},
  organization = {Association for Computational Linguistics}
}
```

## License

This code is licensed under the [MIT](LICENSE) license.

`SentEval` is licensed under the [BSD license](SentEval/LICENSE). The version in
this repo was forked from commit
[`906b34a`](https://github.com/facebookresearch/SentEval/tree/906b34ae5ffbe17a6970947d5dd5e500ff6daf59).

`word-embeddings-benchmark` (web) is licensed under the
[MIT](substring_nli/third_party/web/LICENSE) license. The version
in this repo was forked from
commit[`8fd0489`](https://github.com/kudkudak/word-embeddings-benchmarks/commit/8fd04891a92d313cc3b6956a43f25c9e44022e0e).
