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

arg_parser = argparse.ArgumentParser(description='Preprocess SNLI dataset')

arg_parser.add_argument('--force_reload', action='store_true',
                        help='Whether to reload pickles or not (makes the '
                        'process slower, but ensures data coherence)')
arg_parser.add_argument('--reload_lang', action='store_true',
                        help='Whether to reload pickles or not within Lang (makes the '
                        'process slower, but ensures data coherence)')
arg_parser.add_argument('--min_freq_threshold', type=int, default=2,
                        help='Only words that appear at least this number '
                             'of times will be considered')

SNLI_FIELDS = ['prem_token_ids', 'hypo_token_ids',
               'prem_char_ids', 'hypo_char_ids',
               'label_id', 'pairID']


def main():

    args = arg_parser.parse_args()

    basename = os.path.basename(config.SNLI_TRAIN_PATH)
    filename_no_ext = os.path.splitext(basename)[0]
    train_pickle_path = os.path.join(config.CACHE_PATH, filename_no_ext + '.pkl')
    train = load_or_create(train_pickle_path,
                           pd.read_json,
                           config.SNLI_TRAIN_PATH,
                           lines=True,
                           force_reload=args.force_reload)
    hyps = train['sentence1'].tolist()
    prems = train['sentence2'].tolist()
    all_train_sents = hyps + prems
    lang = Lang(all_train_sents,
                mode='snli',
                min_freq_threshold=args.min_freq_threshold,
                force_reload=args.reload_lang)

    print('Preprocessing training set')
    # New columns must be named the same as SNLI_FIELDS
    train['prem_token_ids'] = train['sentence1'].apply(lang.sent2ids)
    train['hypo_token_ids'] = train['sentence2'].apply(lang.sent2ids)
    train['prem_char_ids'] = train['sentence1'].apply(lang.sent2char_ids)
    train['hypo_char_ids'] = train['sentence2'].apply(lang.sent2char_ids)

    # label_encoder = LabelEncoder()
    # label_encoder.fit(train['gold_label'])
    def label_map(label_str): return config.LABEL2ID[label_str]

    train = train[train['gold_label'] != '-']

    train['label_id'] = train['gold_label'].apply(label_map)

    # We just need a subset of the columns
    train = train[SNLI_FIELDS]

    if not os.path.exists(config.PREPROCESSED_DATA_PATH):
        os.makedirs(config.PREPROCESSED_DATA_PATH)
        print(f'Created {config.PREPROCESSED_DATA_PATH}')

    train.to_json(config.SNLI_TRAIN_PREPROCESSED_PATH,
                  orient='records', lines=True)

    del(train)
    print(f'{config.SNLI_TRAIN_PREPROCESSED_PATH} created')

    print('Preprocessing dev set')
    dev = pd.read_json(config.SNLI_DEV_PATH, lines=True)

    dev['prem_token_ids'] = dev['sentence1'].apply(lang.sent2ids)
    dev['hypo_token_ids'] = dev['sentence2'].apply(lang.sent2ids)
    dev['prem_char_ids'] = dev['sentence1'].apply(lang.sent2char_ids)
    dev['hypo_char_ids'] = dev['sentence2'].apply(lang.sent2char_ids)

    # Remove extraneous labels
    dev = dev[dev['gold_label'] != '-']

    # dev['label_id'] = label_encoder.transform(dev['gold_label'])
    dev['label_id'] = dev['gold_label'].apply(label_map)

    dev = dev[SNLI_FIELDS]

    dev.to_json(config.SNLI_DEV_PREPROCESSED_PATH,
                orient='records',
                lines=True)

    del(dev)
    print(f'{config.SNLI_DEV_PREPROCESSED_PATH} created')

    test = pd.read_json(config.SNLI_TEST_PATH,
                        lines=True)
    test['prem_token_ids'] = test['sentence1'].apply(lang.sent2ids)
    test['hypo_token_ids'] = test['sentence2'].apply(lang.sent2ids)
    test['prem_char_ids'] = test['sentence1'].apply(lang.sent2char_ids)
    test['hypo_char_ids'] = test['sentence2'].apply(lang.sent2char_ids)

    test = test[test['gold_label'] != '-']
    # test['label_id'] = label_encoder.transform(test['gold_label'])
    test['label_id'] = test['gold_label'].apply(label_map)

    test = test[SNLI_FIELDS]

    test.to_json(config.SNLI_TEST_PREPROCESSED_PATH,
                 orient='records',
                 lines=True)
    print(f'{config.SNLI_TEST_PREPROCESSED_PATH} created')


if __name__ == '__main__':
    main()
