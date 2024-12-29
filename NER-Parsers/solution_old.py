import zipfile
from typing import List, Iterable, Set, Tuple
from transformers import AutoTokenizer, AutoModelForTokenClassification

from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding

import re
import json
from typing import Tuple, List, Dict, Any

import torch
import numpy as np

from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


# saved_model_dir = "pretrained.zip"
model_dir = "pretrained"
# with zipfile.ZipFile(saved_model_dir, 'r') as zip_ref:
#     zip_ref.extractall(model_dir)

label2idx = {}
with open(model_dir + "/label2idx.json", "r", encoding="utf-8") as reader:
    label2idx = json.load(reader)

token2idx = {}
with open(model_dir + "/label2idx.json", "r", encoding="utf-8") as reader:
    token2idx = json.load(reader)

idx2label = {}
with open(model_dir + "/idx2label.json", "r", encoding="utf-8") as reader:
    idx2label = json.load(reader)


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

def transform_output(outputs):
    outputs = outputs.detach().cpu().numpy()
    outputs = np.transpose(outputs, (0, 2, 1))
    outputs = outputs.reshape(-1, outputs.shape[-1])
    y_pred = np.argmax(outputs, axis=-1)
    return y_pred

def evaluate_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Iterable[Set[Tuple[int, int, str]]]:
    model.eval()

    results = []
    with torch.no_grad():
        for i, (tokens, labels, mytokens_pos, mytokens) in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc="loop over test batches",
        ):
            device_tokens = tokens.to(device)

            outputs = model(**device_tokens)
            outputs = outputs["logits"].transpose(1, 2)
            results += [transform_output(outputs)]

    return results

class TransformersCollator:
    """
    Transformers Collator that handles variable-size sentences.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        tokenizer_kwargs: Dict[str, Any],
        label_padding_value: int,
    ):
        """
        TransformersCollator class constructor.

        Args:
            tokenizer: the pretrained tokenizer which converts sentence
                to tokens.
            tokenizer_kwargs: the arguments of the tokenizer
            label_padding_value: the padding value for a label

        Returns:
            None
        """
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs

        self.label_padding_value = label_padding_value

    def __call__(
        self,
        batch: List[Tuple[List[str], List[int], List[Tuple[int, int]]]],
    ) -> Tuple[torch.LongTensor, torch.LongTensor, ]:
        """
        Calls transformers' collator.

        Args:
            batch: One batch with sentence and labels.

        Returns:
            (tokens, labels), where `tokens` is sequence of token
                and `labels` is corresponding labels list
        """
        tokens, labels, token_pos = zip(*batch)

        mytokens = self.tokenizer(tokens, **self.tokenizer_kwargs)
        labels = self.encode_labels(mytokens, labels, self.label_padding_value)

        mytokens.pop("offset_mapping")

        return mytokens, labels, token_pos, tokens

    @staticmethod
    def encode_labels(
        tokens: BatchEncoding,
        labels: List[List[int]],
        label_padding_value: int,
    ) -> torch.LongTensor:

        encoded_labels = []

        for doc_labels, doc_offset in zip(labels, tokens.offset_mapping):

            doc_enc_labels = np.ones(len(doc_offset), dtype=int) * label_padding_value
            arr_offset = np.array(doc_offset)

            doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
            encoded_labels.append(doc_enc_labels.tolist())

        return torch.LongTensor(encoded_labels)


class TransformersDataset(torch.utils.data.Dataset):
    """
    Transformers Dataset for NER.
    """

    def __init__(
        self,
        token_seq: List[List[str]],
        label_seq: List[List[str]],
        token_pos: List[List[Tuple[int, int]]]
    ):
        """
        Class constructor.

        Args:
            token_seq: the list of lists contains token sequences.
            label_seq: the list of lists consists of label sequences.

        Returns:
            None
        """
        self.token_seq = token_seq
        self.label_seq = [self.process_labels(labels, label2idx) for labels in label_seq]
        self.token_pos = token_pos

    def __len__(self):
        """
        Returns length of the dataset.

        Args:
            None

        Returns:
            length of the dataset
        """
        return len(self.token_seq)

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[List[str], List[int], List[Tuple[int, int]]]:
        """
        Gets one item for tthe dataset

        Args:
            idx: the index of the particular element in the dataset

        Returns:
            (tokens, labels), where `tokens` is sequence of token in the dataset
                by index `idx` and `labels` is corresponding labels list
        """
        tokens = self.token_seq[idx]
        labels = self.label_seq[idx]
        token_pos = self.token_pos[idx]

        return tokens, labels, token_pos

    @staticmethod
    def process_labels(
        labels: List[str],
        label2idx: Dict[str, int],
    ) -> List[int]:
        """
        Transform list of labels into list of labels' indices.

        Args:
            labels: the list of strings contains the labels
            label2idx: mapping from a label to an index

        Returns:
            ids: the sequence of indices that correspond to labels
        """

        ids = [label2idx[label] for label in labels]

        return ids


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
        # Prepare data for prediction
        test_token_seq = []
        test_token_pos = []
        test_label_seq = []
        for text in texts:
            token_seq, token_pos = handle_text(text)
            if not (token_seq and token_pos):
                continue
            test_token_seq.append(token_seq)
            test_token_pos.append(token_pos)
            test_label_seq.append(["0"] * len(token_seq))

        test_dataset = TransformersDataset(
            token_seq=test_token_seq,
            label_seq=test_label_seq,
            token_pos=test_token_pos
        )
        collator = TransformersCollator(
            tokenizer=self.tokenizer,
            tokenizer_kwargs=self.tokenizer_kwargs,
            label_padding_value=-1,
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collator,
        )

        results = []
        self.model.eval()
        with torch.no_grad():
            for batch in test_dataloader:
                tokens, _, token_pos, original_tokens = batch
                device_tokens = {key: val.to(self.device) for key, val in tokens.items()}

                outputs = self.model(**device_tokens)["logits"]
                predictions = torch.argmax(outputs, dim=-1).cpu().numpy()

                # print([idx2label[str(pred)] for pred in predictions[0]])
                # print(predictions)

                for i, pred_seq in enumerate(predictions):
                    text_results = set()
                    for idx, pred in enumerate(pred_seq[: len(original_tokens[i])]):
                        if pred != 0:
                            start, end = token_pos[i][idx]
                            label = list(label2idx.keys())[list(label2idx.values()).index(pred)]
                            text_results.add((start, end, label))
                    results.append(text_results)

        return results

# sol = Solution()
#
# text = [""]
# with open("./test/1135.txt", "r", encoding="utf-8") as reader:
#     texts = reader.readlines()
# print(sol.predict(texts))




