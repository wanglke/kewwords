# coding=utf-8
import pandas as pd
import numpy as np
from utils import get_corpus
import jieba.analyse




class Textrank(object):
    def __init__(self,data):
        self.data = data

    def get_keywords(self, stopkey, topk=5):
        """
        get the top k keywords
        :param topk:
        :return:
        """
        if isinstance(self.data, list):
            key_list = []
            for text in self.data:
                jieba.analyse.set_stop_words("data/stopWord.txt")
                keywords = jieba.analyse.textrank(text, topK=topk, allowPOS=('n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd'))
                key_list.append(keywords)
            return key_list
        if isinstance(self.data, str):
            jieba.analyse.set_stop_words("data/stopWord.txt")
            keywords = jieba.analyse.textrank(self.data , topK=topk, allowPOS=('n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd'))
            return keywords