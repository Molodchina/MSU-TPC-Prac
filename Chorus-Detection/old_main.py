import numpy as np
import pandas as pd
import nltk
import re
import json
from Levenshtein import ratio
from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import contractions
import warnings
import cmudict
import gensim.downloader as api
from numpy.linalg import norm
from text2vec import SentenceModel

warnings.filterwarnings(action='ignore')

# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')
# nltk.download('punkt')

cmus = cmudict.dict()


def preprocess_lyrics(lyrics):
    # stop_words = set(stopwords.words('english') + stopwords.words('russian'))
    res = []
    for line in lyrics:
        line = re.sub(r'\(.*?\)', '', line)
        line = re.sub(r'\[.*?]', '', line)
        line = re.sub(r'[^\w\s]', '', line).lower()
        line = contractions.fix(line)
        tokens = word_tokenize(line)
        # tokens = [token for token in tokens if token not in stop_words]
        # tagged = nltk.pos_tag(tokens)
        # print(tagged)
        res += [tokens]
    return res


with open('tracks.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
df = pd.DataFrame(data)
df = df.iloc[0:1]
df['processed'] = df['lines'].apply(preprocess_lyrics)


def levenshtein_distance(s1, s2):
    return ratio(s1, s2)

def compute_ssm(lyrics, measure=levenshtein_distance):
    n = len(lyrics)
    ssm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ssm[i][j] = measure(lyrics[i], lyrics[j])
    return ssm

def get_cmus(line: str) -> list[str]:
    return sum(sum([cmus[word] for word in word_tokenize(line)], []), [])

def head_similarity(line1, line2):
    return levenshtein_distance(line1[0:2], line2[0:2])

def tail_similarity(line1, line2):
    return levenshtein_distance(line1[-2:], line2[-2:])

def phonetic_similarity(line1, line2):
    return levenshtein_distance(get_cmus(line1), get_cmus(line2))

def pos_similarity(line1, line2):
    pos1 = nltk.pos_tag(line1)
    pos2 = nltk.pos_tag(line2)
    return levenshtein_distance(pos1, pos2)

def compute_all_ssm(lyrics):
    ssms = {
        'sim_str': compute_ssm(lyrics),
        'sim_head': compute_ssm(lyrics, measure=head_similarity),
        'sim_tail': compute_ssm(lyrics, measure=tail_similarity),
        'sim_phone': compute_ssm(lyrics, measure=phonetic_similarity),
        'sim_pos': compute_ssm(lyrics, measure=pos_similarity)
    }
    return ssms

# word2vec_model = api.load('fasttext-wiki-news-subwords-300')
# flattened_data = [sublist2 for sublist1 in df['processed'] for sublist2 in sublist1]
# word2vec_model = Word2Vec(sentences=flattened_data, vector_size=100, window=5, min_count=1, workers=4) # TODO: vector_size=300,
# sent2vec_model = SentenceModel()


def get_average_word_vector(tokens, model):
    vectors = [model[word] for word in tokens if word in model]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

def create_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def prepare_data(lyrics_data):
    feature_vectors = []
    for lyric_lines in lyrics_data:
        lyric_vectors = [get_average_word_vector(line, word2vec_model) for line in lyric_lines]
        feature_vectors.append(lyric_vectors)
    return np.asarray(feature_vectors, dtype="object")

def label_data(row):
    label = np.zeros(len(row['processed']))
    for chorus_range in row['chorus']:
        begin, end = chorus_range
        label[begin:end] = 1
    return label

def cos_sim(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def sim_str(line1, line2):
    return levenshtein_distance(line1, line2)

def sim_head(line1, line2):
    return levenshtein_distance(line1.split()[:2], line2.split()[:2])

def sim_tail(line1, line2):
    return levenshtein_distance(line1.split()[-2:], line2.split()[-2:])

def sim_phone(line1, line2):
    phon1 = get_cmus(line1)
    phon2 = get_cmus(line2)
    return levenshtein_distance(phon1, phon2)

def sim_pos(line1, line2):
    pos1 = nltk.pos_tag(word_tokenize(line1))
    pos2 = nltk.pos_tag(word_tokenize(line2))
    return levenshtein_distance([p[1] for p in pos1], [p[1] for p in pos2])

def sim_w2v(line1, line2):
    vec1 = np.mean([word2vec_model[word] for word in word_tokenize(line1) if word in word2vec_model], axis=0)
    vec2 = np.mean([word2vec_model[word] for word in word_tokenize(line2) if word in word2vec_model], axis=0)
    return cos_sim(vec1, vec2)

def sim_c2v(line1, line2):
    vec1 = sent2vec_model.encode(line1)
    vec2 = sent2vec_model.encode(line2)
    return cos_sim(vec1, vec2)

# def syllable_count(word):
#     return len([ph for ph in phonetic_transcription(word) if ph[-1].isdigit()])
#
# def sim_syW(line1, line2):
#     syW1 = [syllable_count(word) for word in word_tokenize(line1)]
#     syW2 = [syllable_count(word) for word in word_tokenize(line2)]
#     return Levenshtein.distance(syW1, syW2)
#
# def sim_syL(line1, line2):
#     syL1 = sum([syllable_count(word) for word in word_tokenize(line1)])
#     syL2 = sum([syllable_count(word) for word in word_tokenize(line2)])
#     return abs(syL1 - syL2)

def build_cmp(lyrics):
    for lyric in lyrics:
        n = len(lyric)
        ssm = np.eye(int(n / 4))
        for i in range(0, n - 4, 4):
            lyric1 = np.mean(lyric[i : i + 4], axis=0)
            for j in range(i + 4, n, 4):
                lyric2 = np.mean(lyric[j : j + 4], axis=0)
                ssm[int(i / 4)][int(j / 4)] = ssm[int(j / 4)][int(i / 4)] = cos_sim(lyric1, lyric2)
        return ssm

def if_chorus(target, cur):
    return levenshtein_distance(target, cur) > 0.8

def arin_sym(lyrics):
    results = []
    for lyric in lyrics:
        choruses = []
        n = len(lyric)
        for start in range(0, n):
            for length in range(2, 25):
                chorus = lyric[start:start + length]
                chorus_indices = [[start, start + length - 1]]

                for i in range(start + length, len(lyric) - length + 1):
                    if all(if_chorus(chorus[j], lyric[i + j]) for j in range(length)):
                        chorus_indices.append([i, i + length - 1])

                if len(chorus_indices) > 1:
                    choruses.append((chorus_indices, length))

        filtered_chorus = []
        cur_indices, cur_length = choruses[0]
        for i in range(1, len(choruses)):
            indices, length = choruses[i]
            if len(cur_indices) == len(indices):
                if length > cur_length:
                    cur_indices, cur_length = indices, length
                else:
                    filtered_chorus += [(cur_indices[0], cur_indices[1])]
        if cur_indices != filtered_chorus[-1]:
            filtered_chorus += [(cur_indices[0], cur_indices[1])]

        # Count occurrences of each chorus
        chorus_counts = {}
        for indices in filtered_chorus:
            key = tuple(indices)
            if key in chorus_counts:
                chorus_counts[key] += 1
            else:
                chorus_counts[key] = 1

        # Select the most frequent chorus
        most_frequent_chorus = max(chorus_counts, key=chorus_counts.get)

        # Extract the indices of the most frequent chorus
        selected_chorus_indices = list(most_frequent_chorus)
        results += [selected_chorus_indices]
    return np.array(results)


df['label'] = df.apply(label_data, axis=1)

# X = prepare_data(df['processed'])
# y = to_categorical(df['label'])
# X = X.reshape(X.shape[0], X.shape[1], 1)

# print(build_cmp(X))
# print(df['lines'])
# print(df['chorus'])
# print(word2vec_model.most_similar(positive=["рядом"]))
# print(word2vec_model.wv.most_similar('рядом', topn=5))
# print(arin_sym(df['processed']))
y = arin_sym(df['processed'])
print(y)
f1_scores = []
for pred, target in zip(y, df['chorus']):
    f1 = f1_score(pred, target)
    f1_scores.append(f1)
print(f1_scores)

# input_shape = (X.shape[1], 1)
# model = create_cnn_model(input_shape, num_classes=2)
# model.fit(X, y, epochs=10, batch_size=2)

# predictions = model.predict(X)
# predicted_classes = np.argmax(predictions, axis=1)
#
# accuracy = accuracy_score(df['label'][0], predicted_classes)
# print(f'Accuracy: {accuracy}')
# print(predictions)
