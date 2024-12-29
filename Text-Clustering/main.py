from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import json
import re
import codecs


class Solution:
    def __init__(self):
        self.threshold = 0.35

        self.data_path = "prepared_data.json"
        # with open(self.data_path, "r") as reader:
        #     self.prepared_data = json.load(reader)

        corpus = self.prepare_corpus()
        print(len(corpus))
        self.prepared_data = [(corpus[0], "0")]
        for text in corpus[1:]:
            self.predict_dict(text)

    @staticmethod
    def score(dict1, dict2):
        """
        Calculates the F1 score between two dictionaries.

        Parameters:
            dict1: dict[str, int] - The first dictionary.
            dict2: dict[str, int] - The second dictionary.

        Returns:
            float: The F1 score.
        """
        true_positives = sum(min(dict1.get(key, 0), dict2.get(key, 0)) for key in set(dict1) & set(dict2))
        predicted_positives = sum(dict2.values())
        actual_positives = sum(dict1.values())

        precision = true_positives / predicted_positives if predicted_positives > 0 else 0
        recall = true_positives / actual_positives if actual_positives > 0 else 0

        if precision + recall == 0:
            return 0.0
        return 2. * float(precision * recall) / float(precision + recall)

    @staticmethod
    def prepare(text: str):
        stemmer = SnowballStemmer("russian")
        stop_words = set(stopwords.words("russian"))

        clear_text = re.sub(r'[^\w\s]', '', text).lower()
        tokens = word_tokenize(clear_text)
        tokens = [stemmer.stem(word) for word in tokens]
        tokens = [w.lower() for w in tokens if w not in stop_words]

        counter = dict(Counter(tokens))
        return counter

    @classmethod
    def prepare_corpus(cls):
        dataset_path = "dev-dataset.json"
        data: list = []
        with open(dataset_path, "r") as reader:
            data = json.load(reader)

        return [cls.prepare(obj[0]) for obj in data]

    def new_class(self):
        return str(max(int(new_class) for _, new_class in self.prepared_data) + 1)

    def predict_dict(self, cur_dict: dict) -> str:
        max_score = .0
        max_class = "-1"
        for text_dict, text_class in self.prepared_data:
            cur_score = float(self.score(cur_dict, text_dict))
            if cur_score > max_score:
                max_class = text_class
                max_score = cur_score

        cur_class = max_class
        if max_class == "-1" or max_score < self.threshold:
            cur_class = self.new_class()
        self.prepared_data += [(cur_dict, cur_class)]

        return cur_class

    def predict(self, text: str) -> str:
        cur_dict = self.prepare(text)
        with open(self.data_path, "w", encoding='utf8') as writer:
            writer.write(json.dumps(self.prepared_data, indent=2, ensure_ascii=False))

        count = Counter([text_class for _, text_class in self.prepared_data])
        print(len(dict(count)), count.most_common(10))
        return self.predict_dict(cur_dict)


if __name__ == '__main__':
    sol = Solution()

    dataset_path = "dev-dataset.json"
    data: list = []
    with open(dataset_path, "r") as reader:
        data = json.load(reader)
    print(data[2][1], sol.predict(data[2][0]))
