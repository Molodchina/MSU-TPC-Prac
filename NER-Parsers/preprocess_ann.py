import re
import json
from typing import Dict
from os import listdir
from os.path import isfile, join
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


def remember_ann(
    ann_path: str
) -> Dict[str, str]:
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

    # max_ne_len = max(map(lambda ne: len(ne.split()), ne_list.keys()))
    return ne_list

def read_nerel(
    path: str,
) -> Dict[str, str]:
    files = list(set([join(path, Path(f).stem) for f in listdir(path) if isfile(join(path, f))]))

    ann_suffix = ".ann"

    ne_list = {}
    for file in files:
        ne_list.update(remember_ann(file + ann_suffix))
    return ne_list

train_ne_list = read_nerel("./train/")
train_ne_list.update(read_nerel("./dev/"))


with open("ne_list.json", "w", encoding="utf-8") as writer:
    writer.write(json.dumps(train_ne_list, indent=2, ensure_ascii=False))
