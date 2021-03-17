import jieba_fast as jieba
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from cnkw.utils import get_txtlines, preprocess,jieba_add_words
import numpy as np
import itertools
tqdm.pandas(desc='pandas bar')
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class bertkw(object):
    def __init__(self, data):
        self.data = data

    def max_sum_sim(self,
                    doc_embedding,
                    candidate_embeddings,
                    candidates,
                    word_embeddings,
                    words,
                    top_n,
                    nr_candidates):
        """
        get Max Sum Similarity
        :param doc_embedding:
        :param word_embeddings:
        :param words:
        :param top_n:
        :param nr_candidates:
        :return:
        """
        # Calculate distances and extract keywords
        distances = cosine_similarity(doc_embedding, candidate_embeddings)
        distances_candidates = cosine_similarity(candidate_embeddings,
                                                 candidate_embeddings)

        # Get top_n words as candidates based on cosine similarity
        words_idx = list(distances.argsort()[0][-nr_candidates:])
        words_vals = [candidates[index] for index in words_idx]
        distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

        # Calculate the combination of words that are the least similar to each other
        min_sim = np.inf
        candidate = None
        for combination in itertools.combinations(range(len(words_idx)), top_n):
            sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
            if sim < min_sim:
                candidate = combination
                min_sim = sim

        return [words_vals[idx] for idx in candidate]

    def get_keywords(self, stopkey, topk=5):
        """
        get the top k keywords
        :param topk:
        :return:
        """
        jieba_add_words()
        if isinstance(self.data, list):
            key_list = []
            for text in self.data:
                jieba.analyse.set_stop_words("data/stopWord.txt")
                keywords = jieba.analyse.textrank(text, topK=topk, allowPOS=('n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd'))
                key_list.append(keywords)
            return pd.DataFrame({"id": self.idList, " text": self.data, "keywords": key_list})
        if isinstance(self.data, str):
            doc = ' '.join(jieba.lcut(preprocess(self.data,)))

            n_gram_range = (1, 1)
            count = CountVectorizer(ngram_range=n_gram_range, stop_words=stopkey).fit([doc])
            candidates = count.get_feature_names()

            model = SentenceTransformer(r'xlm-r-distilroberta-base-paraphrase-v1')

            doc_embedding = model.encode([doc])
            candidate_embeddings = model.encode(candidates)

            # top_n = 15
            # distances = cosine_similarity(doc_embedding, candidate_embeddings)
            # keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]
            mss_kws = self.max_sum_sim(doc_embedding=doc_embedding,
                                       word_embeddings=candidate_embeddings,
                                       candidate_embeddings=candidate_embeddings,
                                       candidates=candidates,
                                       words=candidates,
                                       top_n=topk,
                                       nr_candidates=20)
            return mss_kws










# def max_sum_sim(doc_embedding, word_embeddings, words, top_n, nr_candidates):
#     """
#     get Max Sum Similarity
#     :param doc_embedding:
#     :param word_embeddings:
#     :param words:
#     :param top_n:
#     :param nr_candidates:
#     :return:
#     """
#     # Calculate distances and extract keywords
#     distances = cosine_similarity(doc_embedding, candidate_embeddings)
#     distances_candidates = cosine_similarity(candidate_embeddings,
#                                             candidate_embeddings)
#
#     # Get top_n words as candidates based on cosine similarity
#     words_idx = list(distances.argsort()[0][-nr_candidates:])
#     words_vals = [candidates[index] for index in words_idx]
#     distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]
#
#     # Calculate the combination of words that are the least similar to each other
#     min_sim = np.inf
#     candidate = None
#     for combination in itertools.combinations(range(len(words_idx)), top_n):
#         sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
#         if sim < min_sim:
#             candidate = combination
#             min_sim = sim
#
#     return [words_vals[idx] for idx in candidate]



# mss_kws = max_sum_sim(doc_embedding=doc_embedding,
#             word_embeddings=candidate_embeddings,
#             words=candidates,
#             top_n=20,
#             nr_candidates=20)




# def mmr(doc_embedding, word_embeddings, words, top_n, diversity):
#     """
#      Maximal Marginal Relevance
#     :param doc_embedding:
#     :param word_embeddings:
#     :param words:
#     :param top_n:
#     :param diversity:
#     :return:
#     """
#     # Extract similarity within words, and between words and the document
#     word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
#     word_similarity = cosine_similarity(word_embeddings)
#
#     # Initialize candidates and already choose best keyword/keyphras
#     keywords_idx = [np.argmax(word_doc_similarity)]
#     candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]
#
#     for _ in range(top_n - 1):
#         # Extract similarities within candidates and between candidates and selected keywords/phrases
#         candidate_similarities = word_doc_similarity[candidates_idx, :]
#         target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)
#
#         # Calculate MMR
#         mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
#         mmr_idx = candidates_idx[np.argmax(mmr)]
#
#         # Update keywords & candidates
#         keywords_idx.append(mmr_idx)
#         candidates_idx.remove(mmr_idx)
#
#     return [words[idx] for idx in keywords_idx]

# mmr_kws = mmr(doc_embedding=doc_embedding,
#               word_embeddings=candidate_embeddings,
#               words=candidates,
#               top_n=20,
#               diversity=0.8)

# print(mss_kws)
# print(mmr_kws)
# print([s for s in mss_kws if s in mmr_kws])