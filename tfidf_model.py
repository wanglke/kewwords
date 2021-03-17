# coding=utf-8
import pandas as pd
import numpy as np
from utils import get_corpus
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer




class Tfidf(object):
    def __init__(self, idList, data):
        self.idList = idList
        self.data = data

    def get_keywords(self,stopkey,topk=5):
        """
        get the top k keywords
        :param
        :param topk:
        :return:
        """
        corpus = get_corpus(self.data, stopkey)
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(corpus)
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(X)
        word = vectorizer.get_feature_names()
        weight = tfidf.toarray()
        ids, key_list = [], []
        for i in range(len(weight)):
            ids.append(self.idList[i])
            df_word, df_weight = [], []
            for j in range(len(word)):
                df_word.append(word[j])
                df_weight.append(weight[i][j])
            df_word = pd.DataFrame(df_word, columns=['word'])
            df_weight = pd.DataFrame(df_weight, columns=['weight'])
            word_weight = pd.concat([df_word, df_weight], axis=1)
            word_weight = word_weight.sort_values(by="weight", ascending=False)
            keyword = np.array(word_weight['word'])
            word_split = [keyword[x] for x in range(0, topk)]
            word_split = " ".join(word_split)
            key_list.append(word_split)

        return pd.DataFrame({"id": ids, " text": self.data, "keywords": key_list})
