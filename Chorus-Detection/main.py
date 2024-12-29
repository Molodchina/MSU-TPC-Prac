import numpy as np
import pandas as pd
import nltk, json, re
from Levenshtein import ratio
from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
import gensim.downloader as api
from gensim.models import Word2Vec
from text2vec import SentenceModel
from sklearn.metrics import f1_score
# from keras.models import Sequential
# from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import Model, Sequential
from keras.layers import Input, Dense, LSTM, Bidirectional, Concatenate
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import contractions
import warnings
import cmudict
from numpy.linalg import norm


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
        res += [tokens]
    return res


with open('tracks.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
df = pd.DataFrame(data)
df = df.iloc[0:1]
df['processed'] = df['lines'].apply(preprocess_lyrics)


def get_cmus(line):
    return sum(sum([cmus[word] for word in line], []), [])

def cos_sim(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def levenshtein_distance(s1, s2):
    return ratio(s1, s2)


# word2vec_model = api.load('fasttext-wiki-news-subwords-300')
# flattened_data = [sublist2 for sublist1 in df['processed'] for sublist2 in sublist1]
# word2vec_model = Word2Vec(sentences=flattened_data, vector_size=300, window=5, min_count=1, workers=4)
# sent2vec_model = SentenceModel()


def label_data(row):
    label = np.zeros(len(row['processed']))
    for chorus_range in row['chorus']:
        begin, end = chorus_range
        label[begin:end] = 1
    return label

def sim_str(line1, line2):
    return levenshtein_distance(line1, line2)

def sim_head(line1, line2):
    return levenshtein_distance(line1[:2], line2[:2])

def sim_tail(line1, line2):
    return levenshtein_distance(line1[-2:], line2[-2:])

def sim_phone(line1, line2):
    phon1 = get_cmus(line1)
    phon2 = get_cmus(line2)
    return levenshtein_distance(phon1, phon2)

def sim_pos(line1, line2):
    pos1 = nltk.pos_tag(line1)
    pos2 = nltk.pos_tag(line2)
    return levenshtein_distance([p[1] for p in pos1], [p[1] for p in pos2])

def sim_w2v(line1, line2):
    vec1 = np.mean([word2vec_model[word] for word in word_tokenize(line1) if word in word2vec_model], axis=0)
    vec2 = np.mean([word2vec_model[word] for word in word_tokenize(line2) if word in word2vec_model], axis=0)
    return cos_sim(vec1, vec2)

def sim_c2v(line1, line2):
    vec1 = sent2vec_model.encode(line1)
    vec2 = sent2vec_model.encode(line2)
    return cos_sim(vec1, vec2)

inner_size = 5
def compute_ssm(lyric):
    n = len(lyric)
    ssm = np.zeros((n, n, inner_size))
    for i in range(n):
        ssm[i][i] = np.ones(inner_size)
    for i in range(n):
        for j in range(i + 1, n):
            s1 = lyric[i]
            s2 = lyric[j]
            ssm[i][j] = ssm[j][i] = [sim_str(s1, s2), sim_head(s1, s2),
                                     sim_tail(s1, s2), sim_phone(s1, s2), sim_pos(s1, s2)]

    # for i in range(0, 6):
    #     for j in range(n):
    #         print(i, j, ssm[i][j])
    #     print("_____row_____")
    return ssm

def word2vec_features(lyrics, word2vec_model):
    features = []
    for line in lyrics:
        words = nltk.word_tokenize(line)
        word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
        if word_vectors:
            features.append(np.mean(word_vectors, axis=0))
        else:
            features.append(np.zeros(word2vec_model.vector_size))
    return np.array(features)

df['label'] = df.apply(label_data, axis=1)
df['ssm'] = df['processed'].apply(compute_ssm)


def cnn_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    return model


# def build_model(ssm_input_shape):
#     # Input for structural features
#     ssm_input = Input(shape=ssm_input_shape)
#     cnn_features = cnn_model(ssm_input_shape)(ssm_input)
#
#     # Concatenate and pass through Bi-LSTM
#     combined = Concatenate()(cnn_features)
#     lstm = Bidirectional(LSTM(64, return_sequences=True))(combined)
#     output = Dense(1, activation='sigmoid')(lstm)
#
#     model = Model(inputs=ssm_input, outputs=output)
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model


# Generate SSM and Linguistic Features
ssm = df['ssm'][0]
labels = df['label'][0]

# Reshape SSM for CNN input
ssm = np.expand_dims(ssm, axis=-1)

# Split data
# X_train_ssm, X_test_ssm, y_train, y_test = train_test_split(
#     ssm, labels, test_size=0.2, random_state=42
# )

X_train_ssm = ssm
print(X_train_ssm.shape[1:])
exit(0)
ssm_input = Input(shape=X_train_ssm.shape[1:])
cnn_features = cnn_model(X_train_ssm.shape[1:])(ssm_input)

# Build and Train Model
# model = build_model(X_train_ssm.shape[1:])
# model.fit(X_train_ssm, y_train, epochs=10, batch_size=8)


