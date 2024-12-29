import re
import random
from collections import Counter, defaultdict, namedtuple
from typing import Tuple, List, Dict, Any

from os import listdir
from os.path import isfile, join
from pathlib import Path

import torch
import numpy as np

from tqdm import tqdm, trange
import warnings

warnings.filterwarnings("ignore")

def handle_text(
    text: str
) -> Tuple[List[str], List[Tuple[int, int]]]:
    pattern = r'\b\w+\b'
    matches = re.finditer(pattern, text.lower())

    tokens = []
    pos = []
    for match in matches:
        tokens += [match.group(0)]
        pos += [(match.start(), match.end())]
    return tokens, pos


def handle_nerel(
        txt_path: str,
        ann_path: str,
) -> Tuple[List[List[str]], List[List[str]]]:
    if not (isfile(txt_path) and isfile(ann_path)):
        return [], []

    with open(txt_path, "r", encoding="utf-8") as reader:
        txt_lines = reader.readlines()

    with open(ann_path, "r", encoding="utf-8") as reader:
        ann_lines = reader.readlines()

    # Create named entities list
    ne_list = {}
    for ann in ann_lines:
        parts = ann.strip().split()
        if not (len(parts) >= 5 and parts[0].startswith("T")):
            continue

        entity_type = parts[1].strip()

        ne_text = " ".join(parts[4:])
        ne_parts = " ".join(list(map(str.strip, re.sub(r'[^\w]', ' ', ne_text).lower().strip().split())))
        ne_list.setdefault(ne_parts, entity_type)

    max_ne_len = max(map(lambda ne: len(ne.split()), ne_list.keys()))

    # Handle text lines
    cur_tokens = []
    cur_labels = []
    token_seq = []
    label_seq = []
    for line in txt_lines:
        if not line.strip():
            if not (cur_tokens and cur_labels):
                continue
            token_seq += [cur_tokens]
            label_seq += [cur_labels]
            cur_tokens = []
            cur_labels = []
        else:
            clear_line = line.strip()
            cur_tokens, cur_pos = handle_text(clear_line)

            cur_labels = ["0" for i in range(len(cur_tokens))]
            for start in range(len(cur_tokens)):
                for end in range(start + 1, min(len(cur_tokens) + 1, start + max_ne_len + 1)):
                    substr = " ".join(cur_tokens[start:end])
                    if substr in ne_list:
                        for label_idx in range(start, end):
                            cur_labels[label_idx] = ne_list[substr]

            # print(f"{cur_tokens}\t{cur_labels}")

    if cur_tokens and cur_labels:
        token_seq += [cur_tokens]
        label_seq += [cur_labels]

    return token_seq, label_seq


def read_nerel(
    path: str,
    lower: bool = True,
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Prepare data in CoNNL like format.

    Args:
        path:   The path to the files dir (str).
        lower:  Reduce text to lowercase (bool).

    Returns:
        Function returns pair (token_seq, label_seq).
        token_seq: The list of lists. Each internal list is
            a sentence converted into tokens.
        label_seq: The list of lists. All internal lists
            contain tags corresponding to tokens from token_seq.

    """

    token_seq: List[List[str]] = []
    label_seq: List[List[str]] = []

    files = list(set([join(path, Path(f).stem) for f in listdir(path) if isfile(join(path, f))]))

    txt_suffux = ".txt"
    ann_suffix = ".ann"

    for file in files:
        cur_tokens, cur_labels = handle_nerel(file + txt_suffux,
                                              file + ann_suffix)
        token_seq += cur_tokens
        label_seq += cur_labels
    return token_seq, label_seq


train_token_seq, train_label_seq = read_nerel("./train/")
valid_token_seq, valid_label_seq = read_nerel("./dev/")
test_token_seq, test_label_seq   = read_nerel("./test/")


def get_token2idx(
    token_seq: List[List[str]],
    min_count: int,
) -> Dict[str, int]:
    """
    Get mapping from tokens to indices to use with Embedding layer.

    Args:
        token_seq: The list of lists. Each internal list (sentence)
            consists of tokens.
        min_count:  The minimum number of repetitions of
            a token in the corpus.

    Returns:
        Function returns mapping from token to id.
        token2idx: The mapping from token
            to id without "rare" words.

    """

    token2idx: Dict[str, int] = {}
    token2cnt = Counter([token for sentence in token_seq for token in sentence])

    # token2cnt = Counter({k: c for k, c in token2cnt.items() if c >= min_count})
    token2idx["<PAD>"] = 0
    token2idx["<UNK>"] = 1

    current_idx = 2
    for token, count in token2cnt.items():
        if count >= min_count:
            token2idx[token] = current_idx
            current_idx += 1

    return token2idx

token2idx = get_token2idx(train_token_seq, min_count=2)

def get_label2idx(label_seq: List[List[str]]) -> Dict[str, int]:
    """
    Get mapping from labels to indices.

    Args:
        label_seq: The list of lists. Each internal list (sentence)
            consists of labels.

    Returns:
        Function returns mapping from label to id.
        label2idx: The mapping from label to id.

    """

    label2idx: Dict[str, int] = {}
    label_list = set(label for sentence in label_seq for label in sentence)
    label_list = sorted(label_list, key=lambda x: 'A' if x == 'O' else x)

    label2idx = {k: i for i, k in enumerate(label_list)}

    return label2idx

label2idx = get_label2idx(train_label_seq)
idx2label = {val: key for key, val in label2idx.items()}

import json


with open("./label2idx.json", "w") as writer:
    writer.write(json.dumps(label2idx, indent=2))

with open("pretrained/idx2label.json", "w") as writer:
    writer.write(json.dumps(idx2label, indent=2))

with open("./token2idx.json", "w", encoding="utf-8") as writer:
    writer.write(json.dumps(token2idx, indent = 2, ensure_ascii=False))
