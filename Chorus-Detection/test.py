import numpy as np
import pandas as pd
import nltk, json, re
from Levenshtein import ratio
from nltk.tokenize import word_tokenize
import contractions
import warnings
import cmudict
from numpy.linalg import norm

warnings.filterwarnings(action='ignore')


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
# df = df.iloc[0:2]
df['processed'] = df['lines'].apply(preprocess_lyrics)


def get_cmus(line):
    return sum(sum([cmus[word] for word in line], []), [])

def cos_sim(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def levenshtein_distance(s1, s2):
    return ratio(s1, s2)


# word2vec_model = api.load('fasttext-wiki-news-subwords-300')
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

def sym(lyrics, choruses, ids):
    results = []
    sims = []
    for lyric, chorus, track_id in zip(lyrics, choruses, ids):
        parts = []
        for pair in chorus:
            parts += [lyric[pair[0] : pair[1]]]

        base_len = min(map(len, parts))
        for i in range(1, len(parts)):
            if len(parts[i]) > base_len:
                if len(parts[i]) == 2 * base_len:
                    parts += [parts[i][base_len:]]
                parts[i] = parts[i][:base_len]
        sims += [[[[sim_str(s1, s2), sim_head(s1, s2), sim_tail(s1, s2), sim_phone(s1, s2), sim_pos(s1, s2)] for s1, s2 in zip(parts[0], parts[i])] for i in range(1, len(parts))]]
        # print(track_id, sims[-1])
        x = np.array(sims[-1])
        print(track_id, x.shape, x)
        results += [np.mean(x, axis=(1, 0))]
    return sims, results

    # df = pd.DataFrame(res_sims, columns=['sim_str', 'sim_head', 'sim_tail', 'sim_phone', 'sim_pos'])
    # average_metrics = df.mean()
    # return df, average_metrics

print(levenshtein_distance("одиночество", "одиночеств"))
y, vals = sym(df['processed'], df['chorus'], df["track_id"])
# print(y)
print(np.mean(vals, axis=0))


# a = np.array([[1, 2, 3],
#               [4, 5, 6],
#               [7, 8, 9]])
#
# b = np.array([[1, 2, 3],
#               [4, 5, 6],
#               [7, 8, 9]])
#
# c = np.array([a, b])
# print(np.mean(c, axis=(1, 0)))
