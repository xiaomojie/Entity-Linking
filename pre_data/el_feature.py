# encoding: utf-8


import json
import Levenshtein
from embed_util import *
import jieba.analyse
from pre_data_config import *
from rake_nltk import Rake
import os

import sys
reload(sys)
sys.setdefaultencoding("utf-8")



class ElFeature(object):
    """
    提取候选实体与mention相关特征
    """

    def __init__(self, source_doc_path):
        self.source_doc_path = source_doc_path

        self.embed_util = EmbedUtil()
        self.rake = Rake(max_length=1)

    def remove_bad_label(self, candidate_entity_list):
        """
        移除候选实体中标注错误和标注重复的实体
        :param candidate_entity_list:
        :return:
        """
        new_candidate_list = candidate_entity_list[:]
        positive_url = ""
        for entity in candidate_entity_list:
            if entity["flag"] == 1:
                positive_url = entity["wiki_url"]
                break

        mention_url_list = []
        for entity in candidate_entity_list:
            mention_url = entity["wiki_url"]
            # 标注错误则移除
            if entity["flag"] == 0 and entity["wiki_url"] == positive_url:
                new_candidate_list.remove(entity)
                continue

            # 为错误的候选实体且重复出现则从列表中移除
            if entity["flag"] == 0 and mention_url in set(mention_url_list):
                new_candidate_list.remove(entity)
            else:
                mention_url_list.append(mention_url)



        return new_candidate_list


    def process_source_doc(self, result_path):
        """
        处理源文档信息, 处理后的数据格式为
        -DOCSTART- \t 文档mention列表
        文档中mention信息格式为
        mention \t mention所在句子 \t 候选实体信息
        :param result_path:
        :return:
        """

        # 一篇文章中的mention
        doc_mention_list = []
        # 一条句子中的mention
        sent_mention_set = set()
        # 一篇文章的句子
        sent_list = []
        # 一条句子中的单词
        word_list = []
        # 上一条句子中的单词
        last_word_list = []
        # mention所对应的句子
        mention_sent_dict = {}

        count = 0

        with open(self.source_doc_path, "r") as source_file:
            with open(result_path, "w") as result_file:
                for item in source_file:
                    item = item.strip().decode("utf-8")
                    content_split = item.split("\t")

                    # 一篇文章开始标志
                    if content_split[0].__contains__("DOCSTART"):
                        doc_mention_list = []
                        sent_mention_set = set()
                        sent_list = []
                        word_list = []
                        last_word_list = []
                        mention_sent_dict = {}
                        continue

                    # 一篇文章的mention对应所有候选实体信息
                    if content_split[0].__contains__("__json_result__"):
                        count += 1

                        try:
                            json_str = content_split[0].split("__json_result__:")[-1]
                            mention_candidate_json = json.loads(json_str)

                            result_file.write("-DOCSTART-".encode("utf-8") + "\t" + json.dumps(doc_mention_list, ensure_ascii=False).encode("utf-8") + "\n")

                            for mention_item in doc_mention_list:
                                candidate_entity_list = []

                                if mention_item not in mention_candidate_json:
                                    continue

                                for item in mention_candidate_json[mention_item]:
                                    if not isinstance(item, dict):
                                        continue
                                    if item.has_key(mention_item):
                                        item.pop(mention_item)
                                    elif item.has_key(mention_item.replace(" ", "_")):
                                        item.pop(mention_item.replace(" ", "_"))
                                    candidate_entity_list.append(item)

                                if len(mention_sent_dict[mention_item]) > 0 and len(candidate_entity_list) > 0:
                                    # 移除错误和重复实体
                                    new_candidate_list = self.remove_bad_label(candidate_entity_list)
                                    result_file.write(mention_item.encode("utf-8") + "\t" +
                                                      " ".join(mention_sent_dict[mention_item][0]) + "\t"
                                                      + json.dumps(new_candidate_list, ensure_ascii=False).encode("utf-8") + "\n")
                                    mention_sent_dict[mention_item].pop(0)
                        except:
                            pass

                        continue

                    # 句子中出现mention, 标注为NIL的目前不被添加
                    if len(content_split) > 1 and content_split[-1] != "--NME--":
                        mention = content_split[2]
                        doc_mention_list.append(mention)
                        sent_mention_set.add(mention)

                    # 一句话结束标志
                    if content_split[0] == "":
                        # 当前句子长度过短时将其与上一个句子拼接
                        if len(word_list) < 11:
                            word_list = last_word_list + word_list

                        # 存储mention对应的句子
                        if len(sent_mention_set) != 0:
                            for mention_item in sent_mention_set:
                                if mention_sent_dict.has_key(mention_item):
                                    mention_sent_dict[mention_item].append(word_list)
                                else:
                                    mention_sent_dict[mention_item] = [word_list]

                        sent_list.append(word_list)
                        last_word_list = word_list
                        word_list = []
                        sent_mention_set = set()
                        continue

                    word_list.append(content_split[0])

        print "doc count:" + str(count)

    def load_source_word_embed(self, source_result_path, source_word_embed_path):
        """
        加载源文件中单词的词向量
        :param source_result_path: 解析后的源文件路径
        :param source_embed_path: 词向量文件路径
        :return: 当前文件中所有单词的词向量, 格式为word_embed[word] = embed
        """
        # 无词向量时,先生成词向量文件
        if not os.path.exists(source_word_embed_path):
            self.embed_util.filter_file_word_embed(source_result_path, source_word_embed_path)

        self.word_embeds = self.embed_util.get_embedding_from_file(source_word_embed_path)

    def load_source_entity_embed(self, source_result_path, source_entity_embed_path):
        """
        加载源文件中的实体向量
        :param source_result_path:
        :param source_entity_embed_path:
        :return:
        """
        # 无实体向量时,先生成实体向量文件
        if not os.path.exists(source_entity_embed_path):
            self.embed_util.filter_file_entity_embed(source_result_path, source_entity_embed_path)

        self.entity_embeds = self.embed_util.get_embedding_from_file(source_entity_embed_path)


    def cal_feature(self, mention, mention_context, candidate_entity):
        """
        计算mention上下文与候选实体的相关特征值
        :param mention: mention名称
        :param mention_context: mention上下文
        :param candidate_entity: 候选实体
        :return:
        """
        # 候选实体的浏览量
        candidate_pv = candidate_entity["pv"]

        # mention_name与entity_name的jaro_winkler距离
        name_dis = Levenshtein.jaro_winkler(mention, candidate_entity["title"])

        # 筛选summary中的关键词
        summary = candidate_entity["summary"]
        self.rake.extract_keywords_from_text(summary)
        summary_keywords = self.rake.get_ranked_phrases()
        summary_keyword_set = set(summary_keywords)
        if len(summary_keywords) > 20:
            summary_keywords = summary_keywords[:20]
        summary = candidate_entity["summary"].split(" ")

        # 提取summary单词的词向量
        summary_embed_dict = {}
        summary_keyword_embed_dict = {}
        # 保证summary长度在100个以内
        if len(summary) > 100:
            summary = summary[:100]
        for summary_word in summary:
            if not self.word_embeds.has_key(summary_word):
                continue
            summary_embed_dict[summary_word] = self.word_embeds[summary_word]

            if summary_word in summary_keyword_set:
                summary_keyword_embed_dict[summary_word] = self.word_embeds[summary_word]

        # mention上下文与候选实体summary所有单词及summary关键词平均相似度
        word_all_cos_list = []
        word_keyword_cos_list = []
        for context_word in mention_context.split(" "):
            # 无当前词向量时跳过后续计算
            if not self.word_embeds.has_key(context_word):
                continue
            context_word_embed = self.word_embeds[context_word]

            word_key_cos = 0
            for summary_word, summary_embed in summary_keyword_embed_dict.items():
                word_key_cos += self.embed_util.cos_distance(context_word_embed, summary_embed)

            if len(summary_keyword_embed_dict.keys()) != 0:
                word_keyword_cos_list.append(word_key_cos / len(summary_keyword_embed_dict.keys()))

            word_cos = 0
            for summary_word, summary_embed in summary_embed_dict.items():
                word_cos += self.embed_util.cos_distance(context_word_embed, summary_embed)

            if len(summary_embed_dict.keys()) != 0:
                word_all_cos_list.append(word_cos / len(summary_embed_dict.keys()))

        mention_summary_cos = 0
        mention_keyword_cos = 0
        if len(word_all_cos_list) != 0:
            mention_summary_cos = sum(word_all_cos_list) / len(word_all_cos_list)
        if len(word_keyword_cos_list) != 0:
            mention_keyword_cos = sum(word_keyword_cos_list) / len(word_keyword_cos_list)

        # 计算entity向量与mention上下文相似度
        entity_url = candidate_entity["wiki_url"]
        entity_url = entity_url.replace("https://en.wikipedia.org/wiki/", "")
        entity_word_cos = 0
        entity_word_cos_list = []
        if self.entity_embeds.has_key(entity_url):
            entity_embd = self.entity_embeds[entity_url]
            for context_word in mention_context.split(" "):
                # 无当前词向量时跳过后续计算
                if not self.word_embeds.has_key(context_word):
                    continue
                context_word_embed = self.word_embeds[context_word]
                entity_word_cos_list.append(self.embed_util.cos_distance(entity_embd, context_word_embed))

            if len(entity_word_cos_list) != 0:
                entity_word_cos = sum(entity_word_cos_list) / len(entity_word_cos_list)


        feature_dict = {
            "pv": candidate_pv / 1000000.0,
            "name_dis": round(name_dis, 6),
            "mention_summary_cos": round(mention_summary_cos, 6),
            "mention_keyword_cos": round(mention_keyword_cos, 6),
            "entity_word_cos": round(entity_word_cos, 6)
        }
        return feature_dict

    def extract_mention_context(self, mention_name, mention_sent, window_size):
        """
        提取mention窗口内上下文
        :param mention_name: mention
        :param mention_sent: mention所在句子
        :param window_size: 窗口大小
        :return:
        """
        mention_sent_list = mention_sent.split(" ")
        mention_sent_size = len(mention_sent_list)
        mention_index = mention_sent_size / 2
        if mention_name in mention_sent_list:
            mention_index = mention_sent_list.index(mention_name)

        mention_context_list = []
        if mention_index < window_size:
            # 前向窗口
            mention_context_list.extend(mention_sent_list[0:mention_index])

            # 后向窗口
            if mention_index + window_size < mention_sent_size:
                mention_context_list.extend(mention_sent_list[mention_index + 1:mention_index + window_size])
            else:
                mention_context_list.extend(mention_sent_list[mention_index + 1:mention_sent_size])
        else:
            # 前向窗口
            mention_context_list.extend(mention_sent_list[mention_index - window_size:mention_index])

            # 后向窗口
            if mention_index + window_size < mention_sent_size:
                mention_context_list.extend(mention_sent_list[mention_index + 1:mention_index + window_size])
            else:
                mention_context_list.extend(mention_sent_list[mention_index + 1:mention_sent_size])

        return " ".join(mention_context_list)


    def extract_entity_feature(self, source_result_path, mention_candidate_feature_path):
        """
        抽取候选实体相关特征
        :param source_result_path: 解析后的源文件路径
        :param mention_candidate_feature_path: mention与候选实体相关特征文件路径
        :return:
        """
        count = 0
        with open(source_result_path, "r") as souce_candidate_file:
            with open(mention_candidate_feature_path, "w") as candidate_feature_file:
                for item in souce_candidate_file:
                    item = item.strip().decode("utf-8")

                    if count % 100 == 0:
                        print "cal fea:" + str(count)

                    count += 1

                    # 文章开头标志
                    if item.__contains__("DOCSTART"):
                        candidate_feature_file.write(item.encode("utf-8") + "\n")
                        continue

                    mention_name, mention_sent, mention_candidate = item.split("\t")

                    mention_candidate_list = json.loads(mention_candidate)

                    new_candidate_list = []
                    # 计算mention上下文与候选实体的相关特征
                    for mention_candidate in mention_candidate_list:
                        # 取mention窗口内词汇
                        mention_context = self.extract_mention_context(mention_name, mention_sent, window_size=4)

                        feature_dict = self.cal_feature(mention_name, mention_context, mention_candidate)

                        # 将计算特征值添加到候选特征中
                        mention_candidate["feature"] = feature_dict
                        new_candidate_list.append(mention_candidate)

                    candidate_feature_file.write(mention_name.encode("utf-8") + "\t" +
                                                 mention_sent.encode("utf-8") + "\t" +
                                                 json.dumps(new_candidate_list, ensure_ascii=False).encode("utf-8")
                                                 + "\n")



    def controller(self):
        # 源文件经过初步处理后文件路径
        source_result_path = self.source_doc_path + "_result"
        # 源文件词向量文件路径
        source_word_embed_path = self.source_doc_path + "_word_embed"
        # 源文件实体向量文件路径
        source_entity_embed_path = self.source_doc_path + "_entity_embed"
        # mention与候选实体特征值文件
        source_entity_feature_path = self.source_doc_path + "_feature"

        # 预处理源文件
        self.process_source_doc(source_result_path)

        # 加载源文件词向量
        self.load_source_word_embed(source_result_path, source_word_embed_path)
        # 加载源文件实体向量
        self.load_source_entity_embed(source_result_path, source_entity_embed_path)

        # 计算mention与候选实体相关特征值
        self.extract_entity_feature(source_result_path, source_entity_feature_path)


if __name__ == "__main__":
    source_doc_path = "/Users/xie/el_data/wiki_crawl_filter"

    el_fea = ElFeature(source_doc_path)
    el_fea.controller()


