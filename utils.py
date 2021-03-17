# -*- coding: utf-8 -*-
"""
Created on 2021.2.23
@author: wlk
"""
import re
import jieba.posseg




def get_txtlines(file_path):
    """
    read txt file to every line into a list
    :param file_path: file path
    :return: stop words list
    """
    return [w.strip() for w in open(file_path, 'r').readlines()]


def jieba_Segmentation(text,stopwords,pos= ['n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd']):
    """
    use package --jieba do chinese Segmentation select part of speech and remove stop words
    :param text: sentence, type:str
    :param stopwords:stopwords,type:list
    :return:res,type:list
    """
    res = []
    seg = jieba.posseg.cut(text)
    for i in seg:
        if i.word not in stopwords and i.flag in pos:
            res.append(i.word)
    return res

def get_corpus(text,stopkey):
    """
    get_corpus
    :param text: sentence list, type:list
    :return: corpus list, type:list
    """
    res = []
    for i in range(len(text)):
        corpus = jieba_Segmentation(text[i], stopkey)
        corpus = " ".join(corpus)
        res.append(corpus)
    return res

def preprocess(document):
    """
    Remove irregular characters from text
    :param document: txt   type:str
    :return: txt type:str
    """
    string = '''！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏.'''
    text = re.sub(string, "", document)
    document = re.sub(r'@[\w_-]+', '', document)
    document = re.sub(r'-', ' ', document)
    document = re.sub('https?://[^ ]+', '', document)
    document = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w +:\ / \ / \S +)", " ", document).split())
    document = re.sub(r'<[^>]*>', '', document)
    document = re.sub(r'\[(.+)\][(].+[)]', r'\1', document)
    document = document.lower()
    return document

def jieba_add_words():
    """
    jieba_add_words
    :return: None
    """
    for word in get_txtlines('data/add_words.txt'):
        jieba.add_word(word)