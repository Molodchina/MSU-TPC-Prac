import zipfile
from typing import List, Iterable, Set, Tuple, Dict, Any
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from torch.nn.utils.rnn import pad_sequence

import math
import json
import torch
import warnings

warnings.filterwarnings("ignore")

saved_model_dir = "./pretrained.zip"
model_dir = "./pretrained"
with zipfile.ZipFile(saved_model_dir, 'r') as zip_ref:
    zip_ref.extractall(model_dir)

label2idx = {}
with open(model_dir + "/label2idx.json", "r", encoding="utf-8") as reader:
    label2idx = json.load(reader)

idx2label = {}
with open(model_dir + "/idx2label.json", "r", encoding="utf-8") as reader:
    idx2label = json.load(reader)

token2idx = {}
with open(model_dir + "/token2idx.json", "r", encoding="utf-8") as reader:
    token2idx = json.load(reader)


class Solution:
    def __init__(self):
        model_dir = "./pretrained"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer_kwargs = {
            "is_split_into_words":    True,
            "return_offsets_mapping": True,
            "padding":                True,
            "truncation":             True,
            "max_length":             512,
            "return_tensors":         "pt",
        }

    def predict(self, texts: List[str]) -> Iterable[Set[Tuple[int, int, str]]]:
        """
        Predict named entity spans for a list of texts.

        Args:
            texts (List[str]): List of input texts.

        Returns:
            Iterable[Set[Tuple[int, int, str]]]: Predicted spans for each text.
        """
        tokenized_texts = self.tokenizer(list(map(str.lower, texts)), return_offsets_mapping=True)

        start_offsets = [torch.tensor([offset[0] for offset in block], device=self.device)
                         for block in tokenized_texts.offset_mapping]
        end_offsets = [torch.tensor([offset[1] for offset in block], device=self.device)
                       for block in tokenized_texts.offset_mapping]

        padded_start_offset = pad_sequence(start_offsets, batch_first=True, padding_value=0).long()
        padded_end_offset = pad_sequence(end_offsets, batch_first=True, padding_value=0).long()
        padded_input_ids = pad_sequence(
            [torch.tensor(ids, device=self.device) for ids in tokenized_texts.input_ids],
            batch_first=True,
            padding_value=0
        ).long()

        max_len = 128
        output_ids_list = []
        for i in range(math.ceil(padded_input_ids.size(1) / max_len)):
            batch = padded_input_ids[:, i * max_len: (i + 1) * max_len]
            logits = self.model(batch).logits
            output_ids_list.append(torch.argmax(logits, dim=-1))

        padded_output_ids = torch.cat(output_ids_list, dim=1)

        outputs = []
        for i in range(len(padded_output_ids)):
            spans = []
            cur_category = padded_output_ids[i][0]
            cur_start_offset = 0
            cur_end_offset = 0

            for j in range(len(start_offsets[i])):
                if cur_category != padded_output_ids[i][j] and (
                        cur_end_offset != padded_start_offset[i][j] or cur_end_offset == 0
                ):
                    if cur_category != 0:
                        if not (cur_start_offset == 0 and cur_end_offset == 0):
                            spans.append((
                                int(cur_start_offset),
                                int(cur_end_offset),
                                idx2label[str(cur_category.item())]
                            ))
                    cur_category = padded_output_ids[i][j]
                    cur_start_offset = padded_start_offset[i][j]
                cur_end_offset = padded_end_offset[i][j]

        if cur_category != 0 and not (cur_start_offset == 0 and cur_end_offset == 0):
            spans.append((
                int(cur_start_offset),
                int(cur_end_offset),
                idx2label[str(cur_category.item())]
            ))
        outputs.append(set(spans))

        return outputs

# sol = Solution()
#
# text = [""]
# with open("./test/1135.txt", "r", encoding="utf-8") as reader:
#     texts = reader.readlines()
# print(len(sol.predict(texts)[0]))
