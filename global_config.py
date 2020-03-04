# encoding: utf-8

import os
import tensorflow as tf
import numpy as np


class GlobalConfigUtil(object):
    """
    全局模型的配置
    """

    def __init__(self):
        self.load_all_entity_embed()


    def gpu_util(self):
        """
        服务器GPU使用配置情况
        :return:
        """
        # use GPU with ID=5
        os.environ["CUDA_VISIBLE_DEVICES"] = "1, 5"
        gpu_config = tf.ConfigProto()
        # maximun alloc gpu80% of MEM
        gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.8
        # allocate dynamically
        gpu_config.gpu_options.allow_growth = True

        return gpu_config

    def load_all_entity_embed(self):
        """
        加载全量实体向量
        :return:
        """
        entity_dict = {}
        entity_embed = np.load(self.pre_train_entity_embed_path)
        with open(self.pre_train_entity_url_path, "r") as entity_url_file:
            count = 0
            for item in entity_url_file:
                item = item.strip().decode("utf-8")
                url = item.split("\t")[0]
                url = url.replace("en.wikipedia.org/wiki/", "")
                entity_dict[url] = entity_embed[count]

                count += 1

        self.all_entity_embedd = entity_dict

    def load_entity_embedd(self, entity_urls):
        """
        返回urls对应的实体向量
        :param entity_url: 实体的wiki_url
        :return: entity embedding(numpy)
        """
        entity_embedds = []

        for entity_url in entity_urls:
            url = entity_url.replace("https://en.wikipedia.org/wiki/", "")
            if self.all_entity_embedd.has_key(url):
                entity_embedds.append(self.all_entity_embedd[url])
            else:
                entity_embedds.append(np.zeros([self.entity_embedding_dim]))

        return np.array(entity_embedds)


    ########### 全局模型相关配置 ###########

    # 本地模型encode向量维度
    local_rep_dim = 512
    # 实体向量维度
    entity_embedding_dim = 300
    # 类别数目
    num_classes = 2
    # 编码方式lstm 或 gru
    rnn = "lstm"
    # rnn隐藏层大小
    hidden_size = 700
    # 全连接层数
    mlp_metric_layer = 2
    # policy模型的全连接层数
    policy_mlp_layer = 4
    # dropout保留比例
    dropout_keep_prob = 0.8
    # 学习率
    learning_rate = 1e-3
    # 序列中的mention数
    seq_mention_num = 4
    # 候选实体数量
    candidate_entity_num = 5
    # 全局模型总迭代轮次
    global_num_epochs = 10
    # policy模型总迭代轮次
    policy_num_epochs = 10
    # batch大小
    batch_size = 16
    # rnn隐藏层数
    num_layers = 2
    # 间隔batch轮输出一次loss
    print_per_batch = 15
    # 提前结束训练间隔的轮数
    local_require_improvement = 3000


    # 训练数据mention排序后文件路径
    train_rank_path = "/home/xmxie/caoyananGroup/fangzheng/project/data/aida_train/aida_train_rank_mention"
    # 训练数据中间表示按照mention排序后的文件路径
    train_rep_rank_mention = "/home/xmxie/caoyananGroup/fangzheng/project/data/aida_train/aida_train_local_rep_mention_rank.npy"
    # 训练数据group文件
    train_group_path = "/home/xmxie/caoyananGroup/fangzheng/project/data/aida_train/aida_train_testa_cut_group"


    # test_name = "ace2004"
    # # 测试数据mention排序后文件路径
    # test_rank_path = "/home/xmxie/caoyananGroup/fangzheng/project/data/" + test_name + "/" + test_name + "_rank_mention"
    # # 测试数据中间表示按照mention排序后的文件路径
    # test_rep_rank_mention = "/home/xmxie/caoyananGroup/fangzheng/project/data/" + test_name + "/" + test_name + "_local_rep_mention_rank.npy"
    # # 测试数据group文件
    # test_group_path = "/home/xmxie/caoyananGroup/fangzheng/project/data/" + test_name + "/" + test_name + "_cut_group"

    # 测试数据mention排序后文件路径
    test_rank_path = "/home/xmxie/caoyananGroup/fangzheng/project/data/pv_list/pv_rank_mention0"
    # 测试数据中间表示按照mention排序后的文件路径
    test_rep_rank_mention = "/home/xmxie/caoyananGroup/fangzheng/project/data/pv_list/pv_local_rep_mention_rank0.npy"
    # 测试数据group文件
    test_group_path = "/home/xmxie/caoyananGroup/fangzheng/project/data/pv_list/cut_group0"


    # 存储全局模型的路径
    save_global_dir = "/home/xmxie/caoyananGroup/fangzheng/project/data/aida_train/model_save/global_model/"

    # 全量实体向量对应url路径
    pre_train_entity_url_path = "/home/xmxie/caoyananGroup/fangzheng/project/data/word_entity_embed/dict.entity"
    # 全量实体向量文件路径
    pre_train_entity_embed_path = "/home/xmxie/caoyananGroup/fangzheng/project/data/word_entity_embed/entity_embeddings.npy"

    ########### 策略模型相关配置 ###########

    # policy model中action个数
    action_size = 5
    # 先验特征维度
    priori_dim = 50

    # 存储policy模型的路径
    save_policy_dir = "/home/xmxie/caoyananGroup/fangzheng/project/data/aida_train/model_save/policy_model/"


