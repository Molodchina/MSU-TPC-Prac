import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from typing import List, Tuple, Set

# Define the LyricsDataset class
class LyricsDataset(Dataset):
    def __init__(self, tracks, tokenizer, max_length):
        self.tracks = tracks
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        track = self.tracks[idx]
        lyric_lines = track['lines']

        encoding = self.tokenizer.encode_plus(
            ' '.join(lyric_lines),
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

# Define the BiLSTMModel class
class BiLSTMModel(nn.Module):
    def __init__(self, bert_model_name):
        super(BiLSTMModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(self.bert.config.hidden_size, 128, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, 2)  # Output layer for binary classification

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state
        x, _ = self.lstm(x)  # x is now [batch_size, seq_length, hidden_size * 2]
        x = self.fc(x)  # [batch_size, seq_length, 2] (for each token)
        return x  # Return output for all time steps

# Define the Solution class with the detect method
class Solution:
    def __init__(self, model_path, bert_model_name, max_length):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BiLSTMModel(bert_model_name).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.max_length = max_length

    def detect(self, tracks: List[Tuple[List[str], str]]) -> List[Set[Tuple[int, int]]]:
        tracks_data = [{'lines': lines, 'title': title} for lines, title in tracks]
        dataset = LyricsDataset(tracks_data, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        results = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                _, preds = torch.max(outputs, dim=2)

                chorus_positions = set()
                start = None
                for i, pred in enumerate(preds[0]):
                    if pred == 1:
                        if start is None:
                            start = i
                    else:
                        if start is not None:
                            chorus_positions.add((start, i))
                            start = None
                if start is not None:
                    chorus_positions.add((start, len(preds[0])))

                results.append(chorus_positions)

        return results

# Main function for prediction
def main():
    # Initialize the Solution class with the model path and parameters
    model_path = 'model.pth'
    bert_model_name = 'bert-base-uncased'
    max_length = 64  # Adjust as necessary

    solution = Solution(model_path, bert_model_name, max_length)

    # Example tracks data
    tracks = [
        (["line1", "line2", "line3"], "title1"),
        (["line4", "line5", "line6"], "title2"),
    ]

    # Detect chorus positions
    chorus_positions = solution.detect(tracks)
    print(chorus_positions)

if __name__ == "__main__":
    main()
