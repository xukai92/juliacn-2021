import os
import csv

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer


class Vocabulary:
    def __init__(self, words):
        word2token = dict()
        token2word = dict()

        for (token, word) in enumerate(words):
            word2token[word] = token + 1
            token2word[token+1] = word

        self.word2token = word2token
        self.token2word = token2word

    def to_tokens(self, words):
        return [(self.word2token[word.lower()]) for word in words if word.lower() in self.word2token ]

    def to_words(self, tokens):
        return [self.token2word[token] for token in tokens if token in self.token2word]

    @property
    def size(self):
        return len(self.word2token)

def load_data(data_dir, text_only=True):
    "Load ABC news from `data_dir`."
    df_raw = pd.read_csv(os.path.join(data_dir, "abcnews-date-text.csv"), parse_dates=[0], infer_datetime_format=True)
    df = df_raw['headline_text']
    df.index = df_raw['publish_date']
    if not text_only:
        return df
    return df.values.tolist()

def preprocess(texts, filter_on=True, return_dense=True, return_vocab=True):
    "Vectorize texts and convert to bags of words."
    cv_kwargs = {"max_df": 0.7, "min_df": 5} if filter_on else {}
    vectorizer = CountVectorizer(stop_words="english", **cv_kwargs)
    bag_of_tokens = vectorizer.fit_transform(texts)
    if return_dense:
        bag_of_tokens = bag_of_tokens.toarray()
    if not return_vocab:
        return bag_of_tokens
    return bag_of_tokens, Vocabulary(vectorizer.get_feature_names())


text_docs = load_data(".")[::200]
docs, vocab = preprocess_bow(text_docs)

print(len(vocab.word2token), len(text_docs))

doc = []
w = []
for d in range(1_000):
    tokens = vocab.to_tokens(text_docs[d].split(" "))
    doc += [(d + 1) for _ in range(len(tokens))]
    w += tokens

np.save("w.npy", np.array(w))
np.save("doc.npy", np.array(doc))
with open('vocab.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(list(vocab.token2word.values()))