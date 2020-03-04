# coding: utf-8

import time
import json
from datetime import timedelta
import tensorflow.contrib.keras as kr
import numpy as np
import re
from rake_nltk import Rake


PAD = "<PAD>"
SPECIFIC_CHAR = u"[\n.,;()\[\]0123456789]"
FEA_WEIGHT_LIST = [0.291, 0.363, 0.064, 0.072, 0.21]
# fea_priori_dim = 5 * FEA_COPY_NUM
FEA_COPY_NUM = 10

class DataUtil(object):

    def __init__(self):
        self.rake = Rake(min_length=1, max_length=3)


    def get_time_dif(self, start_time):
        """
        获取已使用时间
        :param start_time: 起始时间
        :return:
        """
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))

    def process_file(self, filename, word_to_id, max_length=20):
        """
        处理文件,将文件转换为id表示
        :param filename: 文件名
        :param word_to_id: word_to_id[word] = id
        :param max_length: 每个句子最大词数
        :return: mention数据numpy，候选实体numpy, 标签numpy
        """
        contents, summaries, features, urls, labels = self.read_file(filename) # 应该是mention的contents,entity的summary, entity的fea, label，输入还是一个mention一个entity
        content_id, summary_id = [], []

        for i in range(len(contents)):
            content_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])

        for i in range(len(summaries)):
            summary_id.append([word_to_id[x] for x in summaries[i] if x in word_to_id])

        # 使用keras提供的pad_sequences来将文本pad为固定长度
        mention_pad = kr.preprocessing.sequence.pad_sequences(content_id, max_length)
        entity_pad = kr.preprocessing.sequence.pad_sequences(summary_id, max_length)

        return mention_pad, entity_pad, np.array(features), urls, np.array(labels)

    def read_file(self, filename):
        """
        读取label,句子,实体描述
        :param filename: 文件名
        :return: 句子列表, 实体描述列表, 标签列表
        """
        contexts, summaries, features, urls, labels = [], [], [], [], []

        with open(filename) as f:
            for line in f:
                try:
                    line = line.strip().decode("utf-8")
                    if len(line.split('\t')) == 4:
                        label = line.split('\t')[0].split(" ")[0]

                        feas = [np.float(x.split(':')[1]) for x in line.split("\t")[0].split(" ")[2:]]
                        feas = [fea * weight for fea, weight in zip(feas, FEA_WEIGHT_LIST)]
                        feas = [ele for ele in feas for i in range(FEA_COPY_NUM)]
                        # feas = [sum(feas)]

                        context = line.split('\t')[2]
                        entity_str = line.split('\t')[-1]
                        entity_obj = json.loads(entity_str)

                        summary = entity_obj["summary"]
                        summary = re.sub(SPECIFIC_CHAR, "", summary)
                        self.rake.extract_keywords_from_text(summary)
                        summary_phrases = self.rake.get_ranked_phrases()
                        summary_keywords = [ele for phrase in summary_phrases for ele in phrase.split(" ")]
                        if len(summary_keywords) < 15:
                            summary_keywords.extend(summary.split(" "))

                        url = entity_obj["wiki_url"]

                        if context and summary_keywords:
                            contexts.append(context.split(" "))
                            summaries.append(summary_keywords)
                            features.append(feas)
                            urls.append(url)
                            labels.append(int(label))
                except:
                    pass

        return contexts, summaries, features, urls, labels

    def build_vocab(self, data_dir_list, vocab_dir):
        """
        根据训练集构建词汇表，存储
        :param data_dir_list: 数据表路径list
        :param vocab_dir: 词汇表存储路径
        :param vocab_size:
        :return:
        """
        all_data = []
        for dir_item in data_dir_list:
            contexts, summaries, _,  _, _ = self.read_file(dir_item)

            for content_item in contexts:
                all_data.extend(content_item)

            for summary_item in summaries:
                all_data.extend(summary_item)

        all_data_set = set(all_data)
        open(vocab_dir, mode='w').write('\n'.join(all_data_set).encode("utf-8"))

    def read_vocab(self, vocab_dir):
        """
        读取词汇表,将word转化为id表示
        :param vocab_dir: 词汇表路径
        :return: words列表, word_to_id[word] = word_id
        """
        with open(vocab_dir) as fp:
            words = [item.strip().decode("utf-8") for item in fp]
        word_to_id = dict(zip(words, range(len(words))))

        # 防止词汇表部分词无法充当key,导致max(word_to_id.values) != len(word_to_id) - 1
        word_to_id = dict(zip(word_to_id.keys(), range(len(word_to_id))))

        # 添加一个<PAD>来将所有文本pad为同一长度
        words.append(PAD)
        word_to_id[PAD] = len(word_to_id)

        return words, word_to_id

    def load_word_embdding(self, all_word_dict, word_to_id, dim):
        """
        加载预训练词向量
        :param all_word_dict:
        :param word_to_id:
        :param dim:
        :return:  embeddings np
        """
        embeddings = np.zeros([len(word_to_id), dim])
        for word, id in word_to_id.items():
            if all_word_dict.has_key(word):
                embeddings[id] = all_word_dict[word]

        return embeddings

    def process_global_data(self, data_path, local_rep_path, group_path, seq_len, candidate_entity_num):
        """
        处理全局模型训练数据
        :param data_path: 数据路径
        :param local_rep_path: 本地表示路径
        :param group_path: mention按文档划分的group
        :param seq_len: 文档中mention序列长度
        :param candidate_entity_num: 每个mention对应候选实体数量
        :return: mention(numpy), entity(numpy), label(numpy), feature(numpy), entity_url(list)
        """

        # 文件中所有doc
        all_doc_list = []
        # doc中mention
        doc_mention_list = []
        with open(group_path, "r") as group_file:
            for item in group_file:
                item = item.strip().decode("utf-8")
                # 文章开头标志
                if item.__contains__("DOCSTART"):
                    if len(doc_mention_list) != 0:
                        all_doc_list.append(doc_mention_list)
                    doc_mention_list = []
                    continue
                doc_mention_list.append(int(item))

        if len(doc_mention_list) != 0:
            all_doc_list.append(doc_mention_list)

        entity_start_index = 0
        padding_index_list = []
        for doc_mentions in all_doc_list:
            doc_mention_num = len(doc_mentions)
            doc_group_num = int((doc_mention_num - 1) / seq_len) + 1  # 对一篇文章中的mention每4个一组进行分组
            # 对一个group内的mention进行padding，复制最后一个，论文中group中的mention个数设置为4
            for doc_group_index in range(doc_group_num):
                start_id = doc_group_index * seq_len
                end_id = min((doc_group_index + 1) * seq_len, doc_mention_num)
                # 对group内最后一个mention进行复制padding
                if end_id - start_id < seq_len:
                    tmp_pad_list = [entity_start_index + tmp_pad for tmp_pad in range((end_id - start_id - 1) * candidate_entity_num, (end_id - start_id) * candidate_entity_num)]
                    for time in range(seq_len-(end_id - start_id)):
                        padding_index_list.append(tmp_pad_list)

                entity_start_index += (end_id - start_id) * candidate_entity_num

        # 正例为0, 负例为1
        label_list = []
        feature_list = []
        entity_url_list = []
        with open(data_path, "r") as data_file:
            for item in data_file:
                item = item.strip().decode("utf-8")
                label = item.split("\t")[0].split(" ")[0]
                feas = [float(x.split(':')[1]) for x in item.split('\t')[0].split(" ")[2:]]
                feas = [fea * weight for fea, weight in zip(feas, FEA_WEIGHT_LIST)]
                feas = [ele for ele in feas for i in range(FEA_COPY_NUM)]
                entity_str = item.split("\t")[-1]
                entity_obj = json.loads(entity_str)
                entity_url = entity_obj["wiki_url"]
                entity_url_list.append(entity_url)
                label_list.append(int(label)-1)
                feature_list.append(feas)

        # 加载mention和entity本地表示
        local_rep_np = np.load(local_rep_path)
        mention_rep_list = []
        entity_rep_list = []
        for mention_entity_rep in local_rep_np:
            mention_rep_list.append(mention_entity_rep[0])
            entity_rep_list.append(mention_entity_rep[1])

        pad_mention_rep_list = []
        pad_entity_rep_list = []
        pad_label_list = []
        pad_feature_list = []
        pad_entity_url_list = []

        # 对mention,entity,feature,entity_embedd,label进行padding填充
        last_first = 0
        for pad_indexs in padding_index_list:
            first_index = pad_indexs[0]
            if last_first == first_index:
                pad_mention_rep_list.extend(mention_rep_list[first_index:pad_indexs[-1]+1])

                pad_entity_rep_list.extend(entity_rep_list[first_index:pad_indexs[-1]+1])

                pad_label_list.extend(label_list[first_index:pad_indexs[-1]+1])

                pad_feature_list.extend(feature_list[first_index:pad_indexs[-1]+1])

                pad_entity_url_list.extend(entity_url_list[first_index:pad_indexs[-1]+1])
            else:
                pad_mention_rep_list.extend(mention_rep_list[last_first:first_index])
                pad_mention_rep_list.extend(mention_rep_list[first_index:pad_indexs[-1]+1])

                pad_entity_rep_list.extend(entity_rep_list[last_first:first_index])
                pad_entity_rep_list.extend(entity_rep_list[first_index:pad_indexs[-1]+1])

                pad_label_list.extend(label_list[last_first:first_index])
                pad_label_list.extend(label_list[first_index:pad_indexs[-1]+1])

                pad_feature_list.extend(feature_list[last_first:first_index])
                pad_feature_list.extend(feature_list[first_index:pad_indexs[-1] + 1])

                pad_entity_url_list.extend(entity_url_list[last_first:first_index])
                pad_entity_url_list.extend(entity_url_list[first_index:pad_indexs[-1] + 1])

            last_first = first_index

        # 添加末尾
        pad_mention_rep_list.extend(mention_rep_list[last_first:])
        pad_entity_rep_list.extend(entity_rep_list[last_first:])
        pad_label_list.extend(label_list[last_first:])
        pad_feature_list.extend(feature_list[last_first:])
        pad_entity_url_list.extend(entity_url_list[last_first:])

        # 将标签转换为one-hot表示
        num_class = len(set(pad_label_list))
        y_pad = kr.utils.to_categorical(pad_label_list, num_classes=num_class)

        print("source num:{0}, {1}, {2}, {3}, {4}"\
            .format(len(mention_rep_list), len(entity_rep_list), len(label_list), len(feature_list), len(entity_url_list)))
        print("padding num:{0}".format(len(padding_index_list) * candidate_entity_num))
        print("source padding num:{0}, {1}, {2}, {3}, {4}"\
            .format(len(pad_mention_rep_list), len(pad_entity_rep_list), len(pad_label_list), len(pad_feature_list), len(pad_entity_url_list)))

        return np.array(pad_mention_rep_list), np.array(pad_entity_rep_list), np.array(pad_feature_list), pad_entity_url_list, y_pad


    def batch_iter_local(self, mentions, entities, feas, entity_emebdd, y, candidate_entity_num,
                         batch_size=64, is_random=False, is_rank=False):
        """
        本地模型中生成批次数据
        :param mentions: mention (numpy)
        :param entities: 候选实体 (numpy)
        :param feas: 特征值
        :param entity_emebdd: 实体向量
        :param y: 标签
        :param candidate_entity_num: 候选实体数量
        :param batch_size: 每一批的句子数
        :param is_random: 是否将数据打乱
        :param is_random: 训练数据or测试数据
        :return: 批量数据numpy, 实体描述numpy, 标签numpy
        """
        # 批次数
        data_len = len(mentions)
        num_batch = int((data_len - 1) / (batch_size * candidate_entity_num)) + 1

        # mention_shuffle(shape=(B,S)) 获取到所有非重复mention
        mention_list = [mentions[index] for index in range(data_len) if index % candidate_entity_num == 0] # 在训练集中的mention根据candidate_entity_num数量进行复制，每一行是一个mention一个entity
        mention_shuffle = np.array(mention_list)

        pos_entity_list = []
        neg_entity_list = []
        pos_fea_list = []
        neg_fea_list = []
        pos_embedd_list = []
        neg_embedd_list = []
        pos_y_list = []
        start_index = 0
        end_index = candidate_entity_num
        group_index = 0
        group_num = data_len / candidate_entity_num

        while 1:
            group_index += 1
            if group_index > group_num:
                break
            # 获取一个entity的candidate中pos的位置
            _y_list = list(y[start_index:end_index])
            _pos_index = _y_list.index(min(_y_list))

            pos_entity_list.append([entities[start_index + _pos_index]])
            pos_fea_list.append([feas[start_index + _pos_index]])
            pos_embedd_list.append([entity_emebdd[start_index + _pos_index]])
            if is_rank:
                pos_y_list.append(_pos_index)
            else:
                pos_y_list.append(0)
            # 添加负例
            tmp_neg_list = [entities[start_index + neg_index]
                            for neg_index in range(candidate_entity_num) if neg_index != _pos_index]
            tmp_neg_feas = [feas[start_index + neg_index]
                            for neg_index in range(candidate_entity_num) if neg_index != _pos_index]
            tmp_neg_emebdd = [entity_emebdd[start_index + neg_index]
                              for neg_index in range(candidate_entity_num) if neg_index != _pos_index]
            neg_entity_list.append(tmp_neg_list)
            neg_fea_list.append(tmp_neg_feas)
            neg_embedd_list.append(tmp_neg_emebdd)

            start_index = end_index
            end_index = start_index + candidate_entity_num


        # shape=(B,N,S)
        pos_entity_shuffle = np.array(pos_entity_list)
        neg_entity_shuffle = np.array(neg_entity_list)

        # shape=(B,1)
        pos_fea_shuffle = np.array(pos_fea_list)
        pos_embedd_shuffle = np.array(pos_embedd_list)

        # shape=(B,N)
        neg_fea_shuffle = np.array(neg_fea_list)
        neg_embedd_shuffle = np.array(neg_embedd_list)

        pos_y_shuffle = np.array(pos_y_list)

        # 将数据打乱
        if is_random:
            indices = np.random.permutation(np.arange(data_len/candidate_entity_num))
            mention_shuffle = mention_shuffle[indices]
            pos_entity_shuffle = pos_entity_shuffle[indices]
            neg_entity_shuffle = neg_entity_shuffle[indices]
            pos_fea_shuffle = pos_fea_shuffle[indices]
            neg_fea_shuffle = neg_fea_shuffle[indices]
            pos_embedd_shuffle = pos_embedd_shuffle[indices]
            neg_embedd_shuffle = neg_embedd_shuffle[indices]
            pos_y_shuffle = pos_y_shuffle[indices]

        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len/candidate_entity_num)
            yield mention_shuffle[start_id:end_id], pos_entity_shuffle[start_id:end_id], \
                  neg_entity_shuffle[start_id:end_id], pos_fea_shuffle[start_id:end_id], \
                  neg_fea_shuffle[start_id:end_id], pos_embedd_shuffle[start_id:end_id], \
                  neg_embedd_shuffle[start_id:end_id], pos_y_shuffle[start_id:end_id]

    def batch_iter_global_train(self, mentions, entities, features, entity_embedds, labels, seq_len, candidate_entity_num, batch_size=64):
        """
        全局模型训练中生成批次数据
        :param mentions: mention数据
        :param entities: mention对应候选实体数据
        :param features: mention和候选实体间相关特征
        :param entity_embedds: entity embedding
        :param labels: mention与候选实体对应label
        :param seq_len: 文档中mention序列长度
        :param candidate_entity_num: 每个mention对应候选实体数量
        :param batch_size:
        :return:
        """

        # 批次数
        data_len = len(mentions)
        num_batch = int((data_len - 1) / (batch_size * seq_len * candidate_entity_num)) + 1

        # 所有mention表示
        mention_list = [mentions[index] for index in range(data_len) if index % candidate_entity_num == 0]

        # mention按照序列进行划分
        mention_count = 0
        tmp_mention_list = []
        mention_seq_list = []
        for mention in mention_list:
            mention_count += 1
            tmp_mention_list.append(mention)
            if mention_count % seq_len == 0:
                mention_seq_list.append(tmp_mention_list)
                tmp_mention_list = []

        # 候选实体按照mention划分
        entity_count = 0
        tmp_entity_list = []
        tmp_fea_list = []
        tmp_entity_embedd_list = []
        tmp_label_list = []
        mention_entity_list = []
        mention_entity_fea_list = []
        mention_entity_embedd_list = []
        mention_entity_label_list = []

        for entity_rep, fea, entity_embedd, label in zip(entities, features, entity_embedds, labels):
            entity_count += 1
            tmp_entity_list.append(entity_rep)
            tmp_fea_list.append(fea)
            tmp_entity_embedd_list.append(entity_embedd)
            tmp_label_list.append(label)

            if entity_count % candidate_entity_num == 0:
                mention_entity_list.append(tmp_entity_list)
                mention_entity_fea_list.append(tmp_fea_list)
                mention_entity_embedd_list.append(tmp_entity_embedd_list)
                mention_entity_label_list.append(tmp_label_list)
                tmp_entity_list = []
                tmp_fea_list = []
                tmp_entity_embedd_list = []
                tmp_label_list = []

        # 候选实体按照序列进行划分
        mention_count = 0
        tmp_entity_list = []
        tmp_fea_list = []
        tmp_entity_embedd_list = []
        tmp_label_list = []
        entity_seq_list = []
        fea_seq_list = []
        embedd_seq_list = []
        label_seq_list = []
        for mention_entity, mention_entity_fea, mention_entity_embedd, mention_entity_label in \
            zip(mention_entity_list, mention_entity_fea_list, mention_entity_embedd_list, mention_entity_label_list):

            mention_count += 1

            tmp_label = np.argmax(mention_entity_label, -1).tolist()
            right_index = tmp_label.index(min(tmp_label))

            # 以一定概率生成候选下标
            p_list = [0.05 for i in range(candidate_entity_num)]
            p_list[right_index] = 0.8
            random_index = np.random.choice(candidate_entity_num, p=p_list)

            # 只选择正确的候选
            # random_index = right_index

            select_entity = mention_entity[random_index]
            select_fea = mention_entity_fea[random_index]
            select_entity_embedd = mention_entity_embedd[random_index]
            select_label = mention_entity_label[random_index]

            tmp_entity_list.append(select_entity)
            tmp_fea_list.append(select_fea)
            tmp_entity_embedd_list.append(select_entity_embedd)
            tmp_label_list.append(select_label)

            if mention_count % seq_len == 0:
                entity_seq_list.append(tmp_entity_list)
                fea_seq_list.append(tmp_fea_list)
                embedd_seq_list.append(tmp_entity_embedd_list)
                label_seq_list.append(tmp_label_list)
                tmp_entity_list = []
                tmp_fea_list = []
                tmp_entity_embedd_list = []
                tmp_label_list = []

        mention_seq_np = np.array(mention_seq_list)
        entity_seq_np = np.array(entity_seq_list)
        fea_seq_np = np.array(fea_seq_list)
        embedd_seq_np = np.array(embedd_seq_list)
        label_seq_np = np.array(label_seq_list)

        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len / (seq_len * candidate_entity_num))
            yield mention_seq_np[start_id:end_id], entity_seq_np[start_id:end_id], fea_seq_np[start_id:end_id], \
                  embedd_seq_np[start_id:end_id], label_seq_np[start_id:end_id]


    def batch_iter_policy_train(self, mentions, entities, feas, entity_embedds, labels, seq_len,
                                candidate_entity_num, batch_size=64, is_random=False):
        """
        policy模型训练中生成批次数据
        :param mentions: mention数据
        :param entities: mention对应候选实体数据
        :param feas: mention与候选实体相关特征
        :param entity_embedds: entity embedding
        :param labels: mention与候选实体对应label
        :param seq_len: 文档中mention序列长度
        :param candidate_entity_num: 每个mention对应候选实体数量
        :param batch_size:
        :param is_random:
        :return:
        """

        # 批次数
        data_len = len(mentions)
        num_batch = int((data_len - 1) / (batch_size * seq_len * candidate_entity_num)) + 1

        # 所有mention表示
        mention_list = [mentions[index] for index in range(data_len) if index % candidate_entity_num == 0]

        # mention按照序列进行划分
        mention_count = 0
        tmp_mention_list = []
        mention_seq_list = []
        for mention in mention_list:
            mention_count += 1
            tmp_mention_list.append(mention)
            if mention_count % seq_len == 0:
                mention_seq_list.append(tmp_mention_list)
                tmp_mention_list = []

        # 候选实体按照mention划分
        entity_count = 0
        tmp_entity_list = []
        tmp_label_list = []
        tmp_fea_list = []
        tmp_embedd_list = []
        mention_entity_list = []
        mention_entity_label_list = []
        mention_entity_fea_list = []
        mention_entity_embedd_list = []
        for entity_rep, label, fea, embedd in zip(entities, labels, feas, entity_embedds):
            entity_count += 1
            tmp_entity_list.append(entity_rep)
            tmp_label_list.append(label)
            tmp_fea_list.append(fea)
            tmp_embedd_list.append(embedd)

            if entity_count % candidate_entity_num == 0:
                mention_entity_list.append(tmp_entity_list)
                mention_entity_label_list.append(tmp_label_list)
                mention_entity_fea_list.append(tmp_fea_list)
                mention_entity_embedd_list.append(tmp_embedd_list)
                tmp_entity_list = []
                tmp_label_list = []
                tmp_fea_list = []
                tmp_embedd_list = []

        # 候选实体按照序列进行划分
        mention_count = 0
        tmp_entity_list = []
        tmp_label_list = []
        tmp_fea_list = []
        tmp_embedd_list = []
        entity_seq_list = []
        label_seq_list = []
        fea_seq_list = []
        embedd_seq_list = []
        for mention_entity, mention_entity_label, mention_entity_fea, mention_entity_embedd in \
                zip(mention_entity_list, mention_entity_label_list, mention_entity_fea_list, mention_entity_embedd_list):
            mention_count += 1
            tmp_entity_list.append(mention_entity)
            tmp_label_list.append(mention_entity_label)
            tmp_fea_list.append(mention_entity_fea)
            tmp_embedd_list.append(mention_entity_embedd)
            if mention_count % seq_len == 0:
                entity_seq_list.append(tmp_entity_list)
                label_seq_list.append(tmp_label_list)
                fea_seq_list.append(tmp_fea_list)
                embedd_seq_list.append(tmp_embedd_list)
                tmp_entity_list = []
                tmp_label_list = []
                tmp_fea_list = []
                tmp_embedd_list = []



        # shape=(B,S,H)
        mention_seq_np = np.array(mention_seq_list)
        # shape=(B,S,C,H)
        entity_seq_np = np.array(entity_seq_list)
        # shape=(B,S,C,H)
        label_seq_np = np.array(label_seq_list)
        # shape=(B,S,C,H)
        fea_seq_np = np.array(fea_seq_list)
        # shape=(B,S,C,H)
        embedd_seq_np = np.array(embedd_seq_list)

        # 将数据打乱
        if is_random:
            indices = np.random.permutation(np.arange(data_len/(seq_len*candidate_entity_num)))
            mention_seq_np = mention_seq_np[indices]
            entity_seq_np = entity_seq_np[indices]
            label_seq_np = label_seq_np[indices]
            fea_seq_np = fea_seq_np[indices]
            embedd_seq_np = embedd_seq_np[indices]


        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len / (seq_len * candidate_entity_num))
            yield mention_seq_np[start_id:end_id], entity_seq_np[start_id:end_id], \
                  fea_seq_np[start_id:end_id], embedd_seq_np[start_id:end_id], label_seq_np[start_id:end_id]