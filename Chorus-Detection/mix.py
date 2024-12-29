import numpy as np
import Levenshtein
from nltk.corpus import cmudict
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from gensim.models import Word2Vec, KeyedVectors
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

MODEL_TYPE = 'xlm-roberta-base'
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_TYPE)
print(tokenizer.encode("i love you", return_tensors='pt'))
print(tokenizer.encode("i love not you", return_tensors='pt'))
exit(0)

# Load pre-trained word2vec and context2vec models
word2vec_model = KeyedVectors.load_word2vec_format('path_to_word2vec_model', binary=True)
context2vec_model = KeyedVectors.load_word2vec_format('path_to_context2vec_model', binary=True)

# Load CMU pronunciation dictionary
d = cmudict.dict()

def phonetic_transcription(text):
    words = word_tokenize(text)
    phonemes = []
    for word in words:
        if word.lower() in d:
            phonemes.extend(d[word.lower()][0])
    return phonemes

def normalized_levenshtein(s1, s2):
    return Levenshtein.distance(s1, s2) / max(len(s1), len(s2))

def sim_str(line1, line2):
    return normalized_levenshtein(line1, line2)

def sim_head(line1, line2):
    return normalized_levenshtein(line1.split()[:2], line2.split()[:2])

def sim_tail(line1, line2):
    return normalized_levenshtein(line1.split()[-2:], line2.split()[-2:])

def sim_phone(line1, line2):
    phon1 = phonetic_transcription(line1)
    phon2 = phonetic_transcription(line2)
    return normalized_levenshtein(phon1, phon2)

def sim_pos(line1, line2):
    pos1 = pos_tag(word_tokenize(line1))
    pos2 = pos_tag(word_tokenize(line2))
    return normalized_levenshtein([p[1] for p in pos1], [p[1] for p in pos2])

def sim_w2v(line1, line2):
    vec1 = np.mean([word2vec_model[word] for word in word_tokenize(line1) if word in word2vec_model], axis=0)
    vec2 = np.mean([word2vec_model[word] for word in word_tokenize(line2) if word in word2vec_model], axis=0)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def sim_c2v(line1, line2):
    vec1 = context2vec_model.wv[line1]
    vec2 = context2vec_model.wv[line2]
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def syllable_count(word):
    return len([ph for ph in phonetic_transcription(word) if ph[-1].isdigit()])

def sim_syW(line1, line2):
    syW1 = [syllable_count(word) for word in word_tokenize(line1)]
    syW2 = [syllable_count(word) for word in word_tokenize(line2)]
    return Levenshtein.distance(syW1, syW2)

def sim_syL(line1, line2):
    syL1 = sum([syllable_count(word) for word in word_tokenize(line1)])
    syL2 = sum([syllable_count(word) for word in word_tokenize(line2)])
    return abs(syL1 - syL2)

def compute_ssm(lyrics):
    n = len(lyrics)
    ssm = np.zeros((n, n, 9))
    for i in range(n):
        for j in range(n):
            ssm[i, j, 0] = sim_str(lyrics[i], lyrics[j])
            ssm[i, j, 1] = sim_head(lyrics[i], lyrics[j])
            ssm[i, j, 2] = sim_tail(lyrics[i], lyrics[j])
            ssm[i, j, 3] = sim_phone(lyrics[i], lyrics[j])
            ssm[i, j, 4] = sim_pos(lyrics[i], lyrics[j])
            ssm[i, j, 5] = sim_w2v(lyrics[i], lyrics[j])
            ssm[i, j, 6] = sim_c2v(lyrics[i], lyrics[j])
            ssm[i, j, 7] = sim_syW(lyrics[i], lyrics[j])
            ssm[i, j, 8] = sim_syL(lyrics[i], lyrics[j])
    return ssm

class ChorusDetectionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ChorusDetectionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out)
        return out

class LyricsDataset(Dataset):
    def __init__(self, lyrics, ssm, linguistic_features):
        self.lyrics = lyrics
        self.ssm = ssm
        self.linguistic_features = linguistic_features

    def __len__(self):
        return len(self.lyrics)

    def __getitem__(self, idx):
        return self.ssm[idx], self.linguistic_features[idx]

def main():
    lyrics = [
        "I love you",
        "You love me",
        "We are together",
        "Forever and ever"
    ]

    ssm = compute_ssm(lyrics)
    linguistic_features = [np.mean([word2vec_model[word] for word in word_tokenize(line) if word in word2vec_model], axis=0) for line in lyrics]

    dataset = LyricsDataset(lyrics, ssm, linguistic_features)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = ChorusDetectionModel(input_dim=9, hidden_dim=50, output_dim=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        for ssm_batch, linguistic_batch in dataloader:
            ssm_batch = ssm_batch.float()
            linguistic_batch = linguistic_batch.float()
            inputs = torch.cat((ssm_batch, linguistic_batch), dim=1)
            outputs = model(inputs)
            loss = criterion(outputs, torch.tensor([1]))  # Assuming the label is 1 for chorus
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print("Training complete.")

if __name__ == "__main__":
    main()
