#!/usr/bin/env python

import argparse
import random
import os
import sys

import pandas as pd
import colored_traceback

# This script is supposed to be executed from the project's top-level directory
sys.path.append(os.path.abspath(os.curdir))

from substring_nli.utils.io import load_or_create
from substring_nli.corpus.lang import Lang
import substring_nli.config as config

random.seed(1234)
colored_traceback.add_hook(always=True)

arg_parser = argparse.ArgumentParser(description='Preprocess MultiNLI dataset')

arg_parser.add_argument('--force_reload', action='store_true',
                        help='Whether to reload pickles or not (makes the '
                        'process slower, but ensures data coherence)')
arg_parser.add_argument('--reload_lang', action='store_true',
                        help='Whether to reload pickles or not within Lang (makes the '
                        'process slower, but ensures data coherence)')
arg_parser.add_argument('--min_freq_threshold', type=int, default=2,
                        help='Only words that appear at least this number '
                             'of times will be considered')

MULTINLI_FIELDS = ['prem_token_ids', 'hypo_token_ids',
                   'prem_char_ids', 'hypo_char_ids',
                   'label_id', 'pairID']


def main():

    args = arg_parser.parse_args()

    basename = os.path.basename(config.MULTINLI_TRAIN_PATH)
    filename_no_ext = os.path.splitext(basename)[0]
    train_pickle_path = os.path.join(config.CACHE_PATH, filename_no_ext + '.pkl')

    train = load_or_create(train_pickle_path,
                           pd.read_json,
                           config.MULTINLI_TRAIN_PATH,
                           lines=True,
                           force_reload=args.force_reload)
    hyps = train['sentence1'].tolist()
    prems = train['sentence2'].tolist()
    all_train_sents = hyps + prems
    lang = Lang(all_train_sents,
                min_freq_threshold=args.min_freq_threshold,
                force_reload=args.reload_lang)

    # try:
    #     # We need to save this for creating the embeddings later
    #     # Another option to avoid pickling the whole object would be to just use
    #     # the intermediate dicts created while lang was being created
    #     print('Saving lang object')
    #     save_pickle(config.MULTINLI_LANG_PICKLE_PATH, lang)
    # except AttributeError:
    #     print('Tried pickling an object with a lambda function. Try installing '
    #           'the package `dill` to avoid this error')

    print('Preprocessing training set')
    # New columns must be named the same as MULTINLI_FIELDS
    train['prem_token_ids'] = train['sentence1'].apply(lang.sent2ids)
    train['hypo_token_ids'] = train['sentence2'].apply(lang.sent2ids)
    train['prem_char_ids'] = train['sentence1'].apply(lang.sent2char_ids)
    train['hypo_char_ids'] = train['sentence2'].apply(lang.sent2char_ids)

    # label_encoder = LabelEncoder()
    # label_encoder.fit(train['gold_label'])
    def label_map(label_str): return config.LABEL2ID[label_str]

    # train['label_id'] = label_encoder.transform(train['gold_label'])
    train['label_id'] = train['gold_label'].apply(label_map)

    # We just need a subset of the columns
    train = train[MULTINLI_FIELDS]

    if not os.path.exists(config.PREPROCESSED_DATA_PATH):
        os.makedirs(config.PREPROCESSED_DATA_PATH)
        print(f'Created {config.PREPROCESSED_DATA_PATH}')

    train.to_json(config.MULTINLI_TRAIN_PREPROCESSED_PATH,
                  orient='records', lines=True)

    del(train)
    print(f'{config.MULTINLI_TRAIN_PREPROCESSED_PATH} created')

    print('Preprocessing dev sets')
    dev_matched = pd.read_json(config.MULTINLI_DEV_MATCHED_PATH, lines=True)

    dev_matched['prem_token_ids'] = dev_matched['sentence1'].apply(lang.sent2ids)
    dev_matched['hypo_token_ids'] = dev_matched['sentence2'].apply(lang.sent2ids)
    dev_matched['prem_char_ids'] = dev_matched['sentence1'].apply(lang.sent2char_ids)
    dev_matched['hypo_char_ids'] = dev_matched['sentence2'].apply(lang.sent2char_ids)

    # Remove extraneous labels
    dev_matched = dev_matched[dev_matched['gold_label'] != '-']

    # dev_matched['label_id'] = label_encoder.transform(dev_matched['gold_label'])
    dev_matched['label_id'] = dev_matched['gold_label'].apply(label_map)

    dev_matched = dev_matched[MULTINLI_FIELDS]

    dev_matched.to_json(config.MULTINLI_DEV_MATCHED_PREPROCESSED_PATH,
                        orient='records',
                        lines=True)

    del(dev_matched)
    print(f'{config.MULTINLI_DEV_MATCHED_PREPROCESSED_PATH} created')

    dev_mismatched = pd.read_json(config.MULTINLI_DEV_MISMATCHED_PATH,
                                  lines=True)
    dev_mismatched['prem_token_ids'] = dev_mismatched['sentence1'].apply(lang.sent2ids)
    dev_mismatched['hypo_token_ids'] = dev_mismatched['sentence2'].apply(lang.sent2ids)
    dev_mismatched['prem_char_ids'] = dev_mismatched['sentence1'].apply(lang.sent2char_ids)
    dev_mismatched['hypo_char_ids'] = dev_mismatched['sentence2'].apply(lang.sent2char_ids)

    dev_mismatched = dev_mismatched[dev_mismatched['gold_label'] != '-']
    # dev_mismatched['label_id'] = label_encoder.transform(dev_mismatched['gold_label'])
    dev_mismatched['label_id'] = dev_mismatched['gold_label'].apply(label_map)

    dev_mismatched = dev_mismatched[MULTINLI_FIELDS]

    dev_mismatched.to_json(config.MULTINLI_DEV_MISMATCHED_PREPROCESSED_PATH,
                           orient='records',
                           lines=True)
    print(f'{config.MULTINLI_DEV_MISMATCHED_PREPROCESSED_PATH} created')


if __name__ == '__main__':
    main()
