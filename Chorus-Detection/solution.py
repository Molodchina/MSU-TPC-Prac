from typing import List, Tuple, Set
import numpy as np
import pandas as pd
import re, json
from Levenshtein import ratio
from nltk.tokenize import word_tokenize
# import contractions
from numpy.linalg import norm


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
        if_chorus_identical = [if_chorus(cur, targ) for cur, targ in
                               zip(lyrics[last_section[0]: last_section[1] + 1],
                                   lyrics[pair[0]: pair[1] + 1])]
        if pair[0] <= last_section[1] + 1 and all(if_chorus_identical) and not all(
                ident == 1 for ident in if_chorus_identical):
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

            while current_pos < n:
                if_chorus_res = [if_chorus(cur, targ) for cur, targ in
                                 zip(lyrics[current_pos:current_pos + length], chorus)]
                if all(res for res in if_chorus_res):
                    repetitions += [[current_pos, current_pos + length - 1]]
                    current_pos += length
                else:
                    current_pos += 1

            # print('BEFORE', start, length, len(repetitions), repetitions)
            repetitions = unite_contiguous_sections(repetitions, lyrics)
            # print('AFTER', start, length, len(repetitions), repetitions)

            if len(repetitions) >= 2 and (
                    length > best_chorus_length and len(repetitions) >= len(best_chorus_repetitions)) \
                    or (length == best_chorus_length and len(repetitions) > len(best_chorus_repetitions)):
                # print("NEW")
                best_chorus = chorus
                best_chorus_length = length
                best_chorus_repetitions = repetitions

    return best_chorus, best_chorus_length, best_chorus_repetitions


def arsym(lyrics):
    results = []
    for lyric in lyrics:
        best_chorus, best_chorus_length, best_chorus_repetitions = find_best_chorus(lyric)
        results += [[(chorus[0], chorus[1]) for chorus in best_chorus_repetitions]]
    return results


class Solution:
    def detect(self, tracks: List[Tuple[List[str], str]]) -> List[Set[Tuple[int, int]]]:
        lyrics = []
        for lines, name in tracks:
            lyrics += [preprocess_lyrics(lines)]
        return arsym(lyrics)


# with open('tracks.json', 'r', encoding='utf-8') as file:
#     data = json.load(file)
# df = pd.DataFrame(data)
#
# data = []
# for lines, track_id in zip(df['lines'], df['track_id']):
#     data += [(lines, track_id)]
#
# sol = Solution()
# print(sol.detect(data))
