import re

from . import config


def preprocess(sent):
    URL_RE = re.compile(r'(\w:)*\/\/\S+')
    URL2_RE = re.compile(r'www\.\S+\.\S+')
    NUM_RE = re.compile(r'[\$€+-]?\d+(?:[\.,]\d+)?')
    ASTERISK_BEFORE_RE = re.compile(r'(\*)+(\S+)')
    ASTERISK_AFTER_RE = re.compile(r'(\S+)(\*)+')
    DASH_BEFORE_RE = re.compile(r'\s+(-)(\S+)')
    DOUBLE_DASH_RE = re.compile(r'-+')
    STARTING_SINGLE_QUOTE_RE = re.compile(r"^(')")

    SUBSTITUTION_PATTERNS = [(URL_RE, config.URL_TOKEN),
                             (URL2_RE, config.URL_TOKEN),
                             (NUM_RE, config.NUM_TOKEN),
                             (ASTERISK_BEFORE_RE, r'\1 \2'),
                             (ASTERISK_AFTER_RE, r' \1 \2'),
                             (DASH_BEFORE_RE, r' \1 \2'),
                             (DOUBLE_DASH_RE, r'-'),
                             (STARTING_SINGLE_QUOTE_RE, r'\1 ')]

    for compiled_re, substitution in SUBSTITUTION_PATTERNS:
        sent = compiled_re.sub(substitution, sent)
    return sent
