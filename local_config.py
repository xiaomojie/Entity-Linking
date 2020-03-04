# encoding: utf-8

import os
import tensorflow as tf
import numpy as np

class LocalConfigUtil(object):

    def __init__(self, data_helper):
        self.data_helper = data_helper

        self.load_vocab()
        self.load_all_entity_embed()

    def gpu_util(self):
        """
        服务器GPU使用配置情况
        :return:
        """
        # use GPU with ID=5
        os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
        gpu_config = tf.ConfigProto()
        # maximun alloc gpu80% of MEM
        gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.8
        # allocate dynamically
        gpu_config.gpu_options.allow_growth = True

        return gpu_config

    def load_all_word_embed(self):
        """
        加载全量词向量
        :return:
        """
        word_dict = {}
        word_embed = np.load(self.pre_train_embed_path)
        with open(self.pre_train_embed_word_path, "r") as word_vocab_file:
            count = 0
            for item in word_vocab_file:
                item = item.strip().decode("utf-8")
                word = item.split("\t")[0]
                word_dict[word] = word_embed[count]

                count += 1

        return word_dict

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

    def load_vocab(self):
        """
        加载词典和词向量
        :return:
        """
        # 如果不存在词汇表，重建
        if not os.path.exists(self.vocab_dir):
            self.data_helper.build_vocab(self.vocab_path_list, self.vocab_dir)
        words, word_to_id = self.data_helper.read_vocab(self.vocab_dir)
        self.words = words
        self.word_to_id = word_to_id
        self.vocab_size = len(words)

        self.embeddings = (self.data_helper.load_word_embdding
                           (self.load_all_word_embed(), word_to_id, self.embedding_dim)
                           if self.is_pre_train_embed else None)



    ########### 本地模型相关配置 ###########

    # 输入序列长度
    seq_length = 15
    # rnn隐藏层大小
    local_hidden_size = 512
    # 隐藏层层数
    num_layers = 2
    # 词向量维度
    embedding_dim = 300
    # 先验特征维度
    priori_dim = 50
    # 实体向量维度
    entity_embedding_dim = 300
    # dropout保留比例
    dropout_keep_prob = 0.8
    # 学习率
    learning_rate = 1e-3
    # 是否使用预训练词向量
    is_pre_train_embed = True
    # batch大小
    batch_size = 64
    # 候选实体数量
    candidate_entity_num = 5
    # 候选中正例数量
    positive_num = 1
    # 候选中负例数量
    negative_num = 4
    # 编码方式lstm 或 gru
    rnn = "lstm"
    # 间隔batch轮输出一次loss
    print_per_batch = 5
    # 本地模型总迭代轮次
    local_num_epochs = 10
    # 提前结束训练间隔的轮数
    local_require_improvement = 2000
    # 每组排序的mention数
    rank_mention_num = 4

    # mention表示和entity表示的计算方式
    metric = "mlp"
    # 全连接层的层数
    mlp_metric_layer = 4
    # loss函数
    loss = "hinge"
    # 正负例的margin
    margin = 0.1

    # 全量词向量对应词典路径
    pre_train_embed_word_path = "/home/xmxie/caoyananGroup/fangzheng/project/data/word_entity_embed/dict.word"
    # 全量词向量文件路径
    pre_train_embed_path = "/home/xmxie/caoyananGroup/fangzheng/project/data/word_entity_embed/word_embeddings.npy"

    # 全量实体向量对应url路径
    pre_train_entity_url_path = "/home/xmxie/caoyananGroup/fangzheng/project/data/word_entity_embed/dict.entity"
    # 全量实体向量文件路径
    pre_train_entity_embed_path = "/home/xmxie/caoyananGroup/fangzheng/project/data/word_entity_embed/entity_embeddings.npy"

    # 训练测试文件词典路径
    vocab_dir = "/home/xmxie/caoyananGroup/fangzheng/project/data/aida_train/aida_train_output_vocab"

    # 存储本地模型的路径
    save_local_dir = "/home/xmxie/caoyananGroup/fangzheng/project/data/aida_train/model_save/local_model/"

    # 训练文件路径
    train_path = "/home/xmxie/caoyananGroup/fangzheng/project/data/aida_train/aida_train_testa_cut_candidate"
    # 训练文件对应的group文件
    group_path = "/home/xmxie/caoyananGroup/fangzheng/project/data/aida_train/aida_train_testa_cut_group"
    # 训练数据本地模型中间表示存储路径
    local_rep_train_path = "/home/xmxie/caoyananGroup/fangzheng/project/data/aida_train/aida_train_local_rep.npy"
    # 训练文件候选实体排序文件路径
    train_candidate_rank_path = "/home/xmxie/caoyananGroup/fangzheng/project/data/aida_train/aida_train_rank_candidate"
    # 训练数据中间表示按照候选实体排序后的文件路径
    local_rep_candidate_rank_train_path = "/home/xmxie/caoyananGroup/fangzheng/project/data/aida_train/aida_train_local_rep_candidate_rank.npy"
    # 训练数据mention排序后文件路径
    train_mention_rank_path = "/home/xmxie/caoyananGroup/fangzheng/project/data/aida_train/aida_train_rank_mention"
    # 训练数据中间表示按照mention排序后的文件路径
    local_rep_mention_rank_train_path = "/home/xmxie/caoyananGroup/fangzheng/project/data/aida_train/aida_train_local_rep_mention_rank.npy"


    test_name = "aida_testb"

    # 测试文件路径
    test_path = "/home/xmxie/caoyananGroup/fangzheng/project/data/" + test_name + "/" + test_name + "_cut_candidate"
    # 测试文件对应group文件
    test_group_path = "/home/xmxie/caoyananGroup/fangzheng/project/data/" + test_name + "/" + test_name + "_cut_group"
    # 测试文件中间表示存储路径
    local_rep_test_path = "/home/xmxie/caoyananGroup/fangzheng/project/data/" + test_name + "/" + test_name + "_local_rep.npy"
    # 测试文件候选实体排序文件路径
    test_candidate_rank_path = "/home/xmxie/caoyananGroup/fangzheng/project/data/" + test_name + "/" + test_name + "_rank_candidate"
    # 测试数据中间表示按照候选实体排序后的文件路径
    local_rep_candidate_rank_test_path = "/home/xmxie/caoyananGroup/fangzheng/project/data/" + test_name + "/" + test_name + "_local_rep_candidate_rank.npy"
    # 测试数据mention排序后文件路径
    test_mention_rank_path = "/home/xmxie/caoyananGroup/fangzheng/project/data/" + test_name + "/" + test_name + "_rank_mention"
    # 测试数据中间表示按照mention排序后的文件路径
    local_rep_mention_rank_test_path = "/home/xmxie/caoyananGroup/fangzheng/project/data/" + test_name + "/" + test_name + "_local_rep_mention_rank.npy"


    # # 测试文件路径
    # test_path = "/home/xmxie/caoyananGroup/fangzheng/project/data/pv_list/cut_candidate0"
    # # 测试文件对应group文件
    # test_group_path = "/home/xmxie/caoyananGroup/fangzheng/project/data/pv_list/cut_group0"
    # # 测试文件中间表示存储路径
    # local_rep_test_path = "/home/xmxie/caoyananGroup/fangzheng/project/data/pv_list/pv_local_rep0.npy"
    # # 测试文件候选实体排序文件路径
    # test_candidate_rank_path = "/home/xmxie/caoyananGroup/fangzheng/project/data/pv_list/pv_rank_candidate0"
    # # 测试数据中间表示按照候选实体排序后的文件路径
    # local_rep_candidate_rank_test_path = "/home/xmxie/caoyananGroup/fangzheng/project/data/pv_list/pv_local_rep_candidate_rank0.npy"
    # # 测试数据mention排序后文件路径
    # test_mention_rank_path = "/home/xmxie/caoyananGroup/fangzheng/project/data/pv_list/pv_rank_mention0"
    # # 测试数据中间表示按照mention排序后的文件路径
    # local_rep_mention_rank_test_path = "/home/xmxie/caoyananGroup/fangzheng/project/data/pv_list/pv_local_rep_mention_rank0.npy"

    train_aida = "/home/xmxie/caoyananGroup/fangzheng/project/data/combine_train/combine_train_cut_candidate"
    test_aidaB = "/home/xmxie/caoyananGroup/fangzheng/project/data/aida_test/aida_testb_cut_candidate"
    test_ace2004 = "/home/xmxie/caoyananGroup/fangzheng/project/data/ace2004/ace2004_cut_candidate"
    test_msnbc = "/home/xmxie/caoyananGroup/fangzheng/project/data/msnbc/msnbc_cut_candidate"
    test_clueweb = "/home/xmxie/caoyananGroup/fangzheng/project/data/clueweb/clueweb_cut_candidate"
    test_aquaint = "/home/xmxie/caoyananGroup/fangzheng/project/data/aquaint/aquaint_cut_candidate"
    test_wikipedia = "/home/xmxie/caoyananGroup/fangzheng/project/data/wikipedia/wikipedia_cut_candidate"
    vocab_path_list = [train_aida, test_aidaB, test_ace2004, test_msnbc, test_clueweb, test_aquaint, test_wikipedia]
