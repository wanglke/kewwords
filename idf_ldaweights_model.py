# -*- encoding=utf-8 -*-
from os.path import join, dirname, basename
import re
import pdb
import json
import math
import pkuseg
from cnkw.utils import *
import numpy as np
from cnkw.hyperparameters import Hyperparamters as hp
import time

class idf_ldaweights(object):
    """
    1、以tfidf方法基础
    2、将在文本中相邻的关键词合并，并根据权重进行调整
    3、同时合并较为相似的短语
    4、并结合LDA 模型，寻找突出主题的词汇，增加权重，组合成结果
    """

    def __init__(self):
        self.pos_name = hp.pos_name
        self.pos_exception = hp.pos_exception
        self.stricted_pos_name = hp.stricted_pos_name
        self.redundent_strict_pattern = hp.redundent_strict_pattern
        self.redundent_loose_pattern = hp.redundent_loose_pattern
        self.extra_date_ptn = hp.extra_date_ptn
        self.exception_char_ptn = hp.exception_char_ptn
        self.remove_parentheses_ptn = hp.remove_parentheses_ptn
        self.parentheses = hp.parentheses
        self.redundant_char_ptn = hp.redundant_char_ptn
        self.median_idf = hp.median_idf
        self.idf_dict = hp.idf_dict
        self.seg = pkuseg.pkuseg(postag=True)
        self.phrases_length_control_dict = hp.phrases_length_control_dict
        self.phrases_length_control_none = hp. phrases_length_control_none
        self.pos_combine_weights_dict = hp.pos_combine_weights_dict
        self.stop_words = hp.stop_words

        # file:p(topic|word)
        self.topic_word_weight = hp.topic_word_weight
        self.word_num = len(self.topic_word_weight)
        # file:p(word|topic)
        self.word_topic_weight = hp.word_topic_weight
        self.topic_num = len(self.word_topic_weight)
        self._topic_prominence()


    def get_keywords(self, text, top_n=5, with_weight=False,
                          func_word_num=1, stop_word_num=0,
                          max_phrase_len=25,
                          topic_theta=0.5, allow_pos_weight=True,
                          stricted_pos=True, allow_length_weight=True,
                          allow_topic_weight=True,
                          without_person_name=False,
                          without_location_name=False,
                          remove_phrases_list=None,
                          remove_words_list=None,
                          specified_words=dict(), bias=None):
        """
        抽取一篇文本的关键短语
        :param text: utf-8 编码中文文本
        :param top_n: 选取多少个关键短语返回，默认为 5，若为 -1 返回所有短语
        :param with_weight: 指定返回关键短语是否需要短语权重
        :param func_word_num: 允许短语中出现的虚词个数，stricted_pos 为 True 时无效
        :param stop_word_num: 允许短语中出现的停用词个数，stricted_pos 为 True 时无效
        :param max_phrase_len: 允许短语的最长长度，默认为 25 个字符
        :param topic_theta: 主题权重的权重调节因子，默认0.5，范围（0~无穷）
        :param stricted_pos: (bool) 为 True 时仅允许名词短语出现
        :param allow_pos_weight: (bool) 考虑词性权重，即某些词性组合的短语首尾更倾向成为关键短语
        :param allow_length_weight: (bool) 考虑词性权重，即 token 长度为 2~5 的短语倾向成为关键短语
        :param allow_topic_weight: (bool) 考虑主题突出度，它有助于过滤与主题无关的短语（如日期等）
        :param without_person_name: (bool) 决定是否剔除短语中的人名
        :param without_location_name: (bool) 决定是否剔除短语中的地名
        :param remove_phrases_list: (list) 将某些不想要的短语剔除，使其不出现在最终结果中
        :param remove_words_list: (list) 将某些不想要的词剔除，使包含该词的短语不出现在最终结果中
        :param specified_words: (dict) 行业名词:词频，若不为空，则仅返回包含该词的短语
        :param bias: (int|float) 若指定 specified_words，则可选择定义权重增加值
        :return: 关键短语及其权重
        """
        try:
            # 配置参数
            if without_location_name:
                if 'ns' in self.stricted_pos_name:
                    self.stricted_pos_name.remove('ns')
                if 'ns' in self.pos_name:
                    self.pos_name.remove('ns')
            else:
                if 'ns' not in self.stricted_pos_name:
                    self.stricted_pos_name.append('ns')
                if 'ns' not in self.pos_name:
                    self.pos_name.append('ns')

            if without_person_name:
                if 'nr' in self.stricted_pos_name:
                    self.stricted_pos_name.remove('nr')
                if 'nr' in self.pos_name:
                    self.pos_name.remove('nr')
            else:
                if 'nr' not in self.stricted_pos_name:
                    self.stricted_pos_name.append('nr')
                if 'nr' not in self.pos_name:
                    self.pos_name.append('nr')

            # step0: 清洗文本，去除杂质
            text = preprocess(text)

            # step1: 分句，使用北大的分词器 pkuseg 做分词和词性标注
            sentences_list = split_sentences(text)
            sentences_segs_list = list()
            counter_segs_list = list()
            for sen in sentences_list:
                sen_segs = self.seg.cut(sen)
                sentences_segs_list.append(sen_segs)
                counter_segs_list.extend(sen_segs)

            # step2: 计算词频
            total_length = len(counter_segs_list)
            freq_dict = dict()
            for word_pos in counter_segs_list:
                word, pos = word_pos
                if word in freq_dict:
                    freq_dict[word][1] += 1
                else:
                    freq_dict.update({word: [pos, 1]})

            # step3: 计算每一个词的权重
            sentences_segs_weights_list = list()
            for sen, sen_segs in zip(sentences_list, sentences_segs_list):
                sen_segs_weights = list()
                for word_pos in sen_segs:
                    word, pos = word_pos
                    if pos in self.pos_name:  # 虚词权重为 0
                        if word in self.stop_words:  # 停用词权重为 0
                            weight = 0.0
                        else:
                            if word in specified_words:  # 为词计算权重
                                if bias is None:
                                    weight = freq_dict[word][1] * self.idf_dict.get(
                                        word, self.median_idf) / total_length + 1 / specified_words[word]
                                else:
                                    weight = freq_dict[word][1] * self.idf_dict.get(
                                        word, self.median_idf) / total_length + bias
                            else:
                                weight = freq_dict[word][1] * self.idf_dict.get(
                                    word, self.median_idf) / total_length
                    else:
                        weight = 0.0
                    sen_segs_weights.append(weight)
                sentences_segs_weights_list.append(sen_segs_weights)

            # step4: 通过一定规则，找到候选短语集合，以及其权重
            candidate_phrases_dict = dict()
            for sen_segs, sen_segs_weights in zip(
                    sentences_segs_list, sentences_segs_weights_list):
                sen_length = len(sen_segs)

                for n in range(1, sen_length + 1):  # n-grams
                    for i in range(0, sen_length - n + 1):
                        candidate_phrase = sen_segs[i: i + n]

                        # 由于 pkuseg 的缺陷，日期被识别为 n 而非 t，故删除日期
                        res = self.extra_date_ptn.match(candidate_phrase[-1][0])
                        if res is not None:
                            continue

                        # 找短语过程中需要进行过滤，分为严格、宽松规则
                        if not stricted_pos:
                            rule_flag = self._loose_candidate_phrases_rules(
                                candidate_phrase, func_word_num=func_word_num,
                                max_phrase_len=max_phrase_len,
                                stop_word_num=stop_word_num)
                        else:
                            rule_flag = self._stricted_candidate_phrases_rules(
                                candidate_phrase, max_phrase_len=max_phrase_len)
                        if not rule_flag:
                            continue

                        # 由于 pkuseg 的缺陷，会把一些杂质符号识别为 n、v、adj，故须删除
                        redundent_flag = False
                        for item in candidate_phrase:
                            matched = self.redundent_strict_pattern.search(item[0])
                            if matched is not None:
                                redundent_flag = True
                                break
                            matched = self.redundent_loose_pattern.search(item[0])

                            if matched is not None and matched.group() == item[0]:
                                redundent_flag = True
                                break
                        if redundent_flag:
                            continue

                        # 如果短语中包含了某些不想要的词，则跳过
                        if remove_words_list is not None:
                            unwanted_phrase_flag = False
                            for item in candidate_phrase:
                                if item[0] in remove_words_list:
                                    unwanted_phrase_flag = True
                                    break
                            if unwanted_phrase_flag:
                                continue

                        # 如果短语中没有一个 token 存在于指定词汇中，则跳过
                        if specified_words != dict():
                            with_specified_words_flag = False
                            for item in candidate_phrase:
                                if item[0] in specified_words:
                                    with_specified_words_flag = True
                                    break
                            if not with_specified_words_flag:
                                continue

                        # 条件六：短语的权重需要乘上'词性权重'
                        if allow_pos_weight:
                            start_end_pos = None
                            if len(candidate_phrase) == 1:
                                start_end_pos = candidate_phrase[0][1]
                            elif len(candidate_phrase) >= 2:
                                start_end_pos = candidate_phrase[0][1] + '|' + candidate_phrase[-1][1]
                            pos_weight = self.pos_combine_weights_dict.get(start_end_pos, 1.0)
                        else:
                            pos_weight = 1.0

                        # 条件七：短语的权重需要乘上 '长度权重'
                        if allow_length_weight:
                            length_weight = self.phrases_length_control_dict.get(
                                len(sen_segs_weights[i: i + n]),
                                self.phrases_length_control_none)
                        else:
                            length_weight = 1.0

                        # 条件八：短语的权重需要加上`主题突出度权重`
                        if allow_topic_weight:
                            topic_weight = 0.0
                            for item in candidate_phrase:
                                topic_weight += self.topic_prominence_dict.get(
                                    item[0], self.unk_topic_prominence_value)
                            topic_weight = topic_weight / len(candidate_phrase)
                        else:
                            topic_weight = 0.0

                        candidate_phrase_weight = sum(sen_segs_weights[i: i + n])
                        candidate_phrase_weight *= length_weight * pos_weight
                        candidate_phrase_weight += topic_weight * topic_theta

                        candidate_phrase_string = ''.join([tup[0] for tup in candidate_phrase])
                        if remove_phrases_list is not None:
                            if candidate_phrase_string in remove_phrases_list:
                                continue
                        if candidate_phrase_string not in candidate_phrases_dict:
                            candidate_phrases_dict.update(
                                {candidate_phrase_string: [candidate_phrase,
                                                           candidate_phrase_weight]})

            # step5: 将 overlaping 过量的短语进行去重过滤
            # 尝试了依据权重高低，将较短的短语替代重复了的较长的短语，但效果不好，故删去
            candidate_phrases_list = sorted(
                candidate_phrases_dict.items(),
                key=lambda item: len(item[1][0]), reverse=True)

            de_duplication_candidate_phrases_list = list()
            for item in candidate_phrases_list:
                sim_ratio = mmr_similarity(item, de_duplication_candidate_phrases_list)
                if sim_ratio != 1:
                    item[1][1] = (1 - sim_ratio) * item[1][1]
                    de_duplication_candidate_phrases_list.append(item)

            # step6: 按重要程度进行排序，选取 top_n 个
            candidate_phrases_list = sorted(de_duplication_candidate_phrases_list,
                                            key=lambda item: item[1][1], reverse=True)

            if with_weight:
                if top_n != -1:
                    final_res = [(item[0], item[1][1]) for item in candidate_phrases_list[:top_n]
                                 if item[1][1] > 0]
                else:
                    final_res = [(item[0], item[1][1]) for item in candidate_phrases_list
                                 if item[1][1] > 0]
            else:
                if top_n != -1:
                    final_res = [item[0] for item in candidate_phrases_list[:top_n]
                                 if item[1][1] > 0]
                else:
                    final_res = [item[0] for item in candidate_phrases_list
                                 if item[1][1] > 0]
            return final_res

        except Exception as e:
            print('the text is not legal. \n{}'.format(e))
            return []

    def _topic_prominence(self):
        init_prob_distribution = np.array([self.topic_num for i in range(self.topic_num)])

        topic_prominence_dict = dict()
        for word in self.topic_word_weight:
            conditional_prob_list = list()
            for i in range(self.topic_num):
                if str(i) in self.topic_word_weight[word]:
                    conditional_prob_list.append(self.topic_word_weight[word][str(i)])
                else:
                    conditional_prob_list.append(1e-5)
            conditional_prob = np.array(conditional_prob_list)

            tmp_dot_log_res = np.log2(np.multiply(conditional_prob, init_prob_distribution))
            kl_div_sum = np.dot(conditional_prob, tmp_dot_log_res)  # kl divergence
            topic_prominence_dict.update({word: float(kl_div_sum)})

        tmp_list = [i[1] for i in tuple(topic_prominence_dict.items())]
        max_prominence = max(tmp_list)
        min_prominence = min(tmp_list)
        for k, v in topic_prominence_dict.items():
            topic_prominence_dict[k] = (v - min_prominence) / (max_prominence - min_prominence)

        self.topic_prominence_dict = topic_prominence_dict

        tmp_prominence_list = [item[1] for item in self.topic_prominence_dict.items()]
        self.unk_topic_prominence_value = sum(tmp_prominence_list) / (2 * len(tmp_prominence_list))

    def _loose_candidate_phrases_rules(self, candidate_phrase, max_phrase_len=25, func_word_num=1, stop_word_num=0):
        # 条件一：一个短语不能超过 12个 token
        if len(candidate_phrase) > 12:
            return False

        # 条件二：一个短语不能超过 25 个 char
        if len(''.join([item[0] for item in candidate_phrase])) > max_phrase_len:
            return False

        # 条件三：一个短语中不能出现超过一个虚词
        more_than_one_func_word_count = 0
        for item in candidate_phrase:
            if item[1] in self.pos_exception:
                more_than_one_func_word_count += 1
        if more_than_one_func_word_count > func_word_num:
            return False

        # 条件四：短语的前后不可以是虚词、停用词，短语末尾不可是动词
        if candidate_phrase[0][1] in self.pos_exception:
            return False
        if candidate_phrase[len(candidate_phrase) - 1][1] in self.pos_exception:
            return False
        if candidate_phrase[len(candidate_phrase) - 1][1] in ['v', 'd']:
            return False
        if candidate_phrase[0][0] in self.stop_words:
            return False
        if candidate_phrase[len(candidate_phrase) - 1][0] in self.stop_words:
            return False

        # 条件五：短语中不可以超过规定个数的停用词
        has_stop_words_count = 0
        for item in candidate_phrase:
            if item[0] in self.stop_words:
                has_stop_words_count += 1
        if has_stop_words_count > stop_word_num:
            return False
        return True

    def _stricted_candidate_phrases_rules(self, candidate_phrase, max_phrase_len=25):
        # 条件一：一个短语不能超过 12个 token
        if len(candidate_phrase) > 12:
            return False

        # 条件二：一个短语不能超过 25 个 char
        if len(''.join([item[0] for item in candidate_phrase])) > max_phrase_len:
            return False

        # 条件三：短语必须是名词短语，不能有停用词
        for idx, item in enumerate(candidate_phrase):
            if item[1] not in self.stricted_pos_name:
                return False
            if idx == 0:  # 初始词汇不可以是动词
                if item[1] in ['v', 'vd', 'vx']:  # 动名词不算在内
                    return False
            if idx == len(candidate_phrase) - 1:  # 结束词必须是名词
                if item[1] in ['a', 'ad', 'vd', 'vx', 'v']:
                    return False

        # 条件四：短语中不可以有停用词
        for item in candidate_phrase:
           if item[0] in self.stop_words and item[1] not in self.stricted_pos_name:
               return False
        return True

    # @staticmethod
    # def get_keywords(self, text, top_n=5, with_weight=False,
    #                       func_word_num=1, stop_word_num=0,
    #                       max_phrase_len=25,
    #                       topic_theta=0.5, allow_pos_weight=True,
    #                       stricted_pos=True, allow_length_weight=True,
    #                       allow_topic_weight=True,
    #                       without_person_name=False,
    #                       without_location_name=False,
    #                       remove_phrases_list=None,
    #                       remove_words_list=None,
    #                       specified_words=dict(), bias=None):
    #     return 1



if __name__ == '__main__':
    start = time.time()
    text = "随着信息技术的飞速发展， 人类社会进入数字信息时代。 获取和掌握信息的能力己成为衡量一个国家实力强弱的标志。 一切信息伴随需求不同决定其效益不同，而一切有益信息都是从大量数据中分析出来的。海量数据又随时间持续产生、不断流动、进而扩散形成大数据。大数据不仅用来描述数据的量非常巨大，还突出强调处理数据的速度。所以，大数据成为数据分析领域的前沿技术。数据成为当今每个行业和商业领域的重要因素。人们对于数据的海量挖掘和大量运用，不仅标志着产业生产率的增长和消费者的大量盈余，而且也明确地提示着大数据时代已经到来。数据正成为与物质资产和人力资本同样重要的基础生产要素，大数据的使用成为提高企业竞争力的关键要素。数据成为资产、产业垂直整合、泛互联网化是大数据时代的三大发展趋势。一个国家拥有的数据规模及运用的能力将成为综合国力的重要组成部分，对数据的占有权和控制权将成为陆权、海权、空权之外的国家核心权力。大数据与人类息息相关，越来越多的问题可以通过大数据解决。不仅在数据科学与技术层次，而且在商业模式、产业格局、生态价值与教育层面，大数据都能带来新理念和新思维，包括政府宏观部门、不同的产业界与学术界，甚至个人消费者。大数据与互联网一样，不仅是信息技术领域的革命，更加速企业创新，在全球范围引领社会变革并启动透明政府的发展。大数据正在引发一场思维革命，大数据正在改变人们考察世界的方式方法，以前所未有的速度引起社会、经济、学术、科研、国防、军事等领域的深刻变革。大数据除了将更好的解决商业问题，科技问题，还有各种社会问题，形成以人为本的大数据战略。大数据这一新概念不仅指数据规模庞大，也包括处理和应用数据，是数据对象、技术与应用三者的统一。大数据既可以是如政府部门或企业掌握的数据库这种有限数据集合，也可以是如微博、微信、社交网络上虚拟的无限数据集合。大数据技术包括数据采集、存储、管理、分析挖掘、可视化等技术及其集成。大数据应用是应用大数据技术对各种类型的大数据集合获得有价值信息的行为。充分实现大数据的价值惟有坚持对象、技术、应用三位一体同步发展。大数据是信息技术与各行业领域紧密融合的典型领域，有着旺盛需求和广阔前景。把握机遇需要不断跟踪研究大数据并不断提升对大数据的认知和理解，坚持技术创新与应用创新协同共进同时加快经济社会各领域的大数据开发与利用，推动国家、行业、企业对于数据的应用需求和发展水平进入新的阶段。在大数据时代数据作为一种独立存在的实体，其资产价值越来越突出，日益引起人们的重视。从具体的个人到形形色色的企业，从各国政府到各种组织都可以合法地去收集数据。不论个人还是企业，以及政府等都可以是数据的拥有者。今后个人隐私与数据归属权可能关系越来越少，欧洲民众要求政府公开信息的诉求极其强烈，民众有权向政府申请信息公开。除了涉及国家安全和个人隐私的公共信息外，大部分政府信息都可以公开。大数据主要有三个方面对人类经济社会发展影响巨大，归纳起来：一是能够推动实现巨大经济效益，二是能够推动增强社会管理水平，三是能够推动提高安全保障能力。大数据在政府和公共服务领域的应用可有效推动政务工作开展，提高政府部门的服务效率、决策水平和社会管理水平，产生巨大社会价值。总而言之，大数据将为人们提供强有力的新工具，使人们能更加容易地把握事物规律，更准确地认识世界、预测未来和改造世界。大数据问题涉及范围较广，本文在研究中注重概念分析与文献分析。具体来讲，研究中综合运用了科学技术哲学、社会学等学科的理论知识对大数据进行研究。同时做到问题导向、有的放矢，力求立足本专业理论知识，突出对涉及主题的深层次研究，注意多领域理论知识的综合和统一。本论文既从微观上探讨作为大数据源头与使用者的个体与企业面临的个人隐私风险与商业机密保护问题，又从宏观上分析整个产业乃至国家的大数据治理方针与发展策略，尽量找到实现国家、企业与个人协调发展与动态平衡的路径和方法。本论文通过历史发展的脉络来把握大数据的发展态势，运用科学技术哲学的视角探讨大数据的本质与内涵。以大数据的社会效应为理论和逻辑前提，在把握思维变化和产业发展转型的基础上，从技术、经济、社会等不同层面分析了大数据的动力机制和发展路径，总结了当前主要行业发展方式转型的经验，围绕科技与社会的互动关系梳理大数据与社会各相关产业的融合造成的广泛影响和巨大效益，突出分析了大数据金融产生的一系列重要影响，进而分析其背后蕴含的社会风险，探索相应的社会治理办法，最后就中国积极应对大数据时代、推动中国的大数据发展提出适合中国国情的国家战略设想。"
    kw_model = idf_ldaweights()
    key_phrases = kw_model.get_keywords(text, topic_theta=1)
    print('key_phrases_1topic: ', key_phrases)
    # key_phrases = kw_model.get_keywords(text, topic_theta=0)
    # print('key_phrases_notopic: ', key_phrases)
    # key_phrases = kw_model.get_keywords(text, allow_length_weight=False, topic_theta=0.5, max_phrase_len=8)
    # print('key_phrases_05topic: ', key_phrases)
    end = time.time()

    print(round((end-start)*1000))