import numpy as np
import pandas as pd
import re
import json
from Levenshtein import ratio
from nltk.tokenize import word_tokenize
# import contractions
import warnings
from numpy.linalg import norm

warnings.filterwarnings(action='ignore')


def preprocess_lyrics(lyrics):
    res = []
    for line in lyrics:
        line = re.sub(r'\(.*?\)', '', line)
        line = re.sub(r'\[.*?]', '', line)
        line = re.sub(r'[^\w\s]', '', line).lower()
        # line = contractions.fix(line)
        tokens = word_tokenize(line)
        res += [tokens]
    return res


with open('tracks.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
df = pd.DataFrame(data)
# df = df.iloc[0:1]
# df = df.loc[df['track_id'] == "40053532"] # 555305 135016
df['processed'] = df['lines'].apply(preprocess_lyrics)


def levenshtein_distance(s1, s2):
    return ratio(s1, s2)

def cos_sim(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def if_chorus(target, cur):
    dist = levenshtein_distance(target, cur)
    return 1 if dist == 1. else (-1 if dist > 0.9 else 0)

def unite_contiguous_sections(pairs, lyrics):
    if not pairs:
        return []

    pairs.sort(key=lambda x: x[0])
    united_sections = [pairs[0]]
    for pair in pairs[1:]:
        last_section = united_sections[-1]
        if_chorus_identical = [if_chorus(cur, targ) for cur, targ in zip(lyrics[last_section[0] : last_section[1] + 1],
                                                                              lyrics[pair[0] : pair[1] + 1])]
        if pair[0] <= last_section[1] + 1 and all(if_chorus_identical) and not all(ident == 1 for ident in if_chorus_identical):
            # print('\n'.join(line for line in [' '.join(word for word in arr) for arr in lyrics[last_section[0] : last_section[1] + 1]]))
            # print("-------")
            # print('\n'.join(line for line in [' '.join(word for word in arr) for arr in lyrics[pair[0] : pair[1] + 1]]))
            last_section[1] = max(last_section[1], pair[1])
        else:
            united_sections.append(pair)
    return united_sections

def find_best_chorus(lyrics):
    best_chorus = []
    best_chorus_length = 0
    best_chorus_repetitions = []

    n = len(lyrics)
    for start in range(n):
        for length in range(2, (n - start + 1) // 2):
            chorus = lyrics[start:start + length]
            repetitions = [[start, start + length - 1]]
            current_pos = start + length

            while current_pos + length < n:
                if_chorus_res = [if_chorus(cur, targ) for cur, targ in zip(lyrics[current_pos:current_pos + length], chorus)]
                if all(res for res in if_chorus_res):
                    repetitions += [[current_pos, current_pos + length - 1]]
                    current_pos += length
                else:
                    current_pos += 1

            # print('BEFORE', start, length, len(repetitions), repetitions)
            repetitions = unite_contiguous_sections(repetitions, lyrics)
            # print('AFTER', start, length, len(repetitions), repetitions)

            # if len(repetitions) >= 2 and (length > best_chorus_length and len(repetitions) >= len(best_chorus_repetitions))\
            #         or (length == best_chorus_length and len(repetitions) > len(best_chorus_repetitions)):
            if len(repetitions) >= 2 and (
                    length > best_chorus_length and len(repetitions) >= len(best_chorus_repetitions)) \
                    or (len(repetitions) > len(best_chorus_repetitions)):
                # print("NEW")
                best_chorus = chorus
                best_chorus_length = length
                best_chorus_repetitions = repetitions
    return best_chorus, best_chorus_length, best_chorus_repetitions

def arin_sym(lyrics):
    results = []
    for lyric in lyrics:
        best_chorus, best_chorus_length, best_chorus_repetitions = find_best_chorus(lyric)
        results += [best_chorus_repetitions]
    return results

y = arin_sym(df['processed'])
# print(y)

def expand_sections(pairs):
    if not pairs:
        return [], 0

    max_value = max(max(pair) for pair in pairs)
    result = [0] * (max_value + 1)
    count = 0
    for start, end in pairs:
        for i in range(start, end + 1):
            result[i] = 1
            count += 1
    return result, count

def intersect_lists_of_pairs(list1, list2):
    expand1, len1 = expand_sections(list1)
    expand2, len2 = expand_sections(list2)
    result = len([True for i in range(min(len(expand1), len(expand2))) if expand1[i] == expand2[i] == 1])
    return result, len1, len2


f1_scores = []
for pred, target, track_id in zip(y, df['chorus'], df['track_id']):
    inter, pred_amount, target_amount = intersect_lists_of_pairs(pred, target)
    if not inter:
        # print(track_id)
        # print(pred)
        f1_scores += [0.]
        continue
    p = inter / pred_amount
    r = inter / target_amount
    f1_score = 2 * p * r / (p + r)
    f1_scores += [f1_score]
    # if f1_score < 0.5:
    #     print(track_id)
# print(sorted(f1_scores))
print(np.mean(f1_scores))
