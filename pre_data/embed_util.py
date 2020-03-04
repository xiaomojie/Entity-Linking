# encoding: utf-8


import json
import re
import numpy as np
from pre_data_config import *


SPECIFIC_CHAR = u"[\n.,;()\[\]0123456789]"

class EmbedUtil(object):

    def load_word_embed(self):
        """
        加载词向量
        :return:
        """
        word_dict = {}
        word_embed = np.load(config_word_embedding_path)
        with open(config_word_vocab_path, "r") as word_vocab_file:
            count = 0
            for item in word_vocab_file:
                item = item.strip().decode("utf-8")
                word = item.split("\t")[0]
                word_dict[word] = word_embed[count]

                count += 1

        return word_dict

    def load_entity_embed(self):
        """
        加载实体向量
        :return:
        """
        entity_embed = np.load(config_entity_embed)

        entity_dict = {}
        with open(config_entity_url_path, "r") as entity_url_file:
            count = 0
            for item in entity_url_file:
                item = item.strip().decode("utf-8")
                url = item.split("\t")[0]
                url = url.replace("en.wikipedia.org/wiki/", "")
                entity_dict[url] = entity_embed[count]

                count += 1

        return entity_dict

    def load_stop_words(self, stop_word_path):
        """
        加载停用词表
        :param stop_word_path:
        :return: 停用词set
        """
        stop_word_set = set()
        with open(stop_word_path, "r") as stop_word_file:
            for item in stop_word_file:
                item = item.strip().decode("utf-8")
                stop_word_set.add(item)

        return stop_word_set

    def filter_file_word_embed(self, source_path, source_word_embed_path):
        """
        从全量词向量中筛选出源文件中出现的词向量
        :param source_path: 源文件路径
        :param source_word_embed_path: 当前文档的词向量文件
        :return:
        """
        word_embd_dict = self.load_word_embed()

        with open(source_path, "r") as source_file:
            with open(source_word_embed_path, "w") as embed_file:
                all_word_set = set()
                for item in source_file:
                    item = item.strip().decode("utf-8")

                    # 文章开头标志
                    if item.__contains__("DOCSTART"):
                        continue

                    mention_name, mention_sent, mention_candidate = item.split("\t")
                    mention_candidate = json.loads(mention_candidate)

                    # 添加mention单词
                    all_word_set.add(mention_name)
                    for word in mention_name.split(" "):
                        all_word_set.add(word)

                    # 添加mention上下文单词
                    for word in mention_sent.split(" "):
                        all_word_set.add(word)

                    # 添加候选实体单词
                    for entity in mention_candidate:
                        entity_name = entity["title"]
                        summary = entity["summary"]
                        summary = re.sub(SPECIFIC_CHAR, "", summary)

                        all_word_set.add(entity_name)
                        for word in entity_name.split("_"):
                            all_word_set.add(word)

                        for word in summary.split(" "):
                            all_word_set.add(word)

                print "word num:" + str(len(all_word_set))

                # 遍历全量词向量文件,查找源文件单词对应的词向量
                for word in all_word_set:
                    if word_embd_dict.has_key(word):
                        embed = [str(item) for item in word_embd_dict[word]]
                        embed_file.write(word.encode("utf-8") + " " + " ".join(embed) + "\n")

    def filter_file_entity_embed(self, source_path, source_entity_embed_path):
        """
        从全量词向量中筛选出源文件中出现的实体向量
        :param source_path: 源文件路径
        :param source_entity_embed_path: 当前文档的实体向量文件
        :return:
        """
        entity_embd_dict = self.load_entity_embed()

        doc_entity_dict = {}
        with open(source_path, "r") as source_file:
            with open(source_entity_embed_path, "w") as embed_file:
                for item in source_file:
                    item = item.strip().decode("utf-8")

                    # 文章开头标志
                    if item.__contains__("DOCSTART"):
                        continue

                    mention_name, mention_sent, mention_candidate = item.split("\t")
                    mention_candidate = json.loads(mention_candidate)

                    for entity in mention_candidate:
                        url = entity["wiki_url"]
                        url = url.replace("https://en.wikipedia.org/wiki/", "")
                        if entity_embd_dict.has_key(url):
                            doc_entity_dict[url] = entity_embd_dict[url]

                print "entity num:" + str(len(doc_entity_dict.keys()))
                for word, embed in doc_entity_dict.items():
                    embed = [str(element) for element in embed]
                    embed_file.write(word.encode("utf-8") + " " + " ".join(embed) + "\n")


    def get_embedding_from_file(self, source_embed_path):
        """
        从文件中获取词向量
        :param source_embed_path: 词向量文件
        :return: dict类型 dict[word] = vector
        """
        vector_dict = {}
        with open(source_embed_path, "r") as embedding_file:
            for item in embedding_file:
                item = item.decode("utf-8").strip()

                name = item.split(" ")[0]
                vector = [float(element) for element in item.split(" ")[1:]]

                if len(vector) != 300:
                    print "error"
                vector_dict[name] = vector
        return vector_dict


    def cos_distance(self, vector1, vector2):
        """
        余弦距离
        :param vector1:
        :param vector2:
        :return:
        """
        cos = 0.0
        vector1_norm = np.linalg.norm(vector1)
        vector2_norm = np.linalg.norm(vector2)
        if len(vector1) == len(vector2) and len(vector1) > 0 \
                and vector1_norm != 0 and vector2_norm != 0:
            cos = np.dot(vector1, vector2) / (vector1_norm * vector2_norm)

        return cos


if __name__ == "__main__":
    embed_util = EmbedUtil()
