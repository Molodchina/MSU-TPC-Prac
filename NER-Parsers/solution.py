import re
from typing import Tuple, List, Dict, Set, Iterable
import json

class Solution:
    def __init__(self):
        ne_list_path = "ne_list.json"
        self.ne_list = {}
        with open(ne_list_path, "r", encoding="utf-8") as reader:
            self.ne_list = json.load(reader)

        self.max_ne_len = max(map(lambda ne: len(ne.split()), self.ne_list.keys()))

    @staticmethod
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

    def process_text(self, text) -> Set[Tuple[int, int, str]]:
        cur_tokens, cur_pos = self.handle_text(text)

        predictions = set()
        for start in range(len(cur_tokens)):
            for end in range(start + 1, min(len(cur_tokens) + 1, start + self.max_ne_len + 1)):
                substr = " ".join(cur_tokens[start:end])
                if substr in self.ne_list:
                    predictions.add((cur_pos[start][0], cur_pos[end - 1][1], self.ne_list[substr]))

        return predictions

    def predict(self, texts: List[str]) -> Iterable[Set[Tuple[int, int, str]]]:
        return [self.process_text(text) for text in texts]


# sol = Solution()
#
# text = [""]
# with open("./test/1135.txt", "r", encoding="utf-8") as reader:
#     texts = reader.readlines()
# print(sol.predict(texts))
