# coding: utf-8

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from local_config import *
from data_util import *
import os
import time


class LocalModel(object):
    def __init__(self, config):
        self.config = config

        # 输入的数据
        self.input_mention = tf.placeholder(tf.int32, [None, self.config.seq_length], name="input_x")

        self.positive_entities = tf.placeholder(tf.int32, [None, self.config.positive_num, self.config.seq_length],
                                                name="positive_entities")

        self.pos_entity_fea = tf.placeholder(tf.float32,
                                             [None, self.config.positive_num, self.config.priori_dim],
                                             name="pos_entity_fea")

        self.pos_entity_embedd = tf.placeholder(tf.float32,
                                                [None, self.config.positive_num, self.config.entity_embedding_dim],
                                                name="pos_entity_embedd")

        self.negative_entities = tf.placeholder(tf.int32, [None, self.config.negative_num, self.config.seq_length],
                                                name="negative_entities")

        self.neg_entity_fea = tf.placeholder(tf.float32, [None, self.config.negative_num, self.config.priori_dim],
                                             name="neg_entity_fea")

        self.neg_entity_embedd = tf.placeholder(tf.float32,
                                                [None, self.config.negative_num, self.config.entity_embedding_dim],
                                                name="neg_entity_embedd")

        self.input_y = tf.placeholder(tf.int64, [None], name="input_y")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        self.rnn_model()

    def rnn_model(self):
        """
        构造本地表示网络
        :return:
        """

        # lstm核
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(self.config.local_hidden_size, state_is_tuple=True)

        # gru核
        def gru_cell():
            return tf.contrib.rnn.GRUCell(self.config.local_hidden_size)

        # 为每一个rnn核后面加一个dropout层
        def dropout():
            if (self.config.rnn == 'lstm'):
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        # rnn网络输出
        def rnn_output(input):
            cells = [dropout() for _ in range(self.config.num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            _output, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=input, dtype=tf.float32)
            return _output

        # 不使用预训练词向量
        if not self.config.is_pre_train_embed:
            embedding = tf.get_variable(
                name="embedding", dtype=tf.float32,
                shape=[self.config.vocab_size, self.config.embedding_dim]
            )
        # 使用预训练词向量
        else:
            embedding = tf.Variable(
                self.config.embeddings, name="embedding",
                dtype=tf.float32, trainable=False)

        mention_context_embedding = tf.nn.embedding_lookup(embedding, self.input_mention) # 都是上下文的
        pos_entities_embedding = tf.nn.embedding_lookup(embedding, self.positive_entities) # 都是上下文的
        neg_entities_embedding = tf.nn.embedding_lookup(embedding, self.negative_entities) # 都是上下文的

        # 两个lstm网络共享变量
        with tf.variable_scope("sent_encode"):
            # mention上下文编码
            _context_output = rnn_output(mention_context_embedding)
            context_last = _context_output[:, -1, :] # 取最后一个时刻的隐藏层输出

        pos_entity_list = [] # 保存了该mention对应的多个pos candidate entity的lstm之后的编码
        neg_entity_list = [] # 保存了该mention对应的多个neg candidate entity的lstm之后的编码
        with tf.variable_scope("sent_encode", reuse=True):
            # 正例实体编码
            for positive_index in range(self.config.positive_num):
                _pos_entity_output = rnn_output(pos_entities_embedding[:, positive_index, :])
                pos_desc_last = _pos_entity_output[:, -1, :]
                pos_entity_list.append(pos_desc_last)
            # 负例实体编码
            for negative_index in range(self.config.negative_num):
                _neg_entity_output = rnn_output(neg_entities_embedding[:, negative_index, :])
                neg_desc_last = _neg_entity_output[:, -1, :]
                neg_entity_list.append(neg_desc_last)

        # shape=(B,H)
        self.mention_context_rep = context_last # mention的lstm编码
        # shape=(B, N, H) N指的是pos/neg的数量
        self.pos_entity_reps = tf.stack(pos_entity_list, axis=1) # pos can 的lstm编码
        self.neg_entity_reps = tf.stack(neg_entity_list, axis=1) # pos can 的lstm编码

        pos_entity_concate = tf.concat([self.pos_entity_reps, self.pos_entity_fea, self.pos_entity_embedd], axis=-1) # entity context, entity fea, entity embedding
        neg_entity_concate = tf.concat([self.neg_entity_reps, self.neg_entity_fea, self.neg_entity_embedd], axis=-1)

        # mention和正负例候选的相似度
        with tf.name_scope('Metric_Layer'):
            mention_norm = tf.nn.l2_normalize(tf.expand_dims(self.mention_context_rep, 1), 2)
            pos_entity_norm = tf.nn.l2_normalize(pos_entity_concate, 2)
            neg_entity_norm = tf.nn.l2_normalize(neg_entity_concate, 2)
            if self.config.metric == 'cosine':
                metric_p = tf.reduce_sum(mention_norm * pos_entity_norm, axis=2, name='cos_sim_pos')
                metric_n = tf.reduce_sum(mention_norm * neg_entity_norm, axis=2, name='cos_sim_neg')
                self.pos_metric = metric_p
                self.neg_metric = metric_n
            elif self.config.metric == 'mlp':
                metric_p = tf.concat([tf.tile(mention_norm, [1, self.config.positive_num, 1]), pos_entity_norm], axis=2)
                metric_n = tf.concat([tf.tile(mention_norm, [1, self.config.negative_num, 1]), neg_entity_norm], axis=2)
                for l_size in ([128, 64, 32, 16][:self.config.mlp_metric_layer] + [1]):
                    metric_p = tf.layers.dense(inputs=metric_p, units=l_size,
                                               name='output_layer_%d' % l_size,
                                               kernel_initializer=tf.glorot_uniform_initializer(),
                                               bias_initializer=tf.constant_initializer(0.0),
                                               activation=tf.nn.softplus)
                    metric_n = tf.layers.dense(inputs=metric_n, units=l_size,
                                               name='output_layer_%d' % l_size,
                                               kernel_initializer=tf.glorot_uniform_initializer(),
                                               bias_initializer=tf.constant_initializer(0.0),
                                               activation=tf.nn.softplus,
                                               reuse=True)
                # shape=(B,P)
                self.pos_metric = tf.squeeze(metric_p, squeeze_dims=[-1])

                # metric_p = self.pos_entity_fea * metric_p
                # self.pos_metric = tf.squeeze(metric_p, squeeze_dims=[-1])

                # shape=(B,N)
                self.neg_metric = tf.squeeze(metric_n, squeeze_dims=[-1])

                # metric_n = self.neg_entity_fea * metric_n
                # self.neg_metric = tf.squeeze(metric_n, squeeze_dims=[-1])

            else:
                raise NotImplementedError

        # loss
        with tf.name_scope('Loss_layer'):
            metric_p = tf.tile(tf.expand_dims(self.pos_metric, axis=2), [1, 1, self.config.negative_num])
            metric_n = tf.tile(tf.expand_dims(self.neg_metric, axis=1), [1, self.config.positive_num, 1])
            delta = metric_n - metric_p

            # loss per mention-pos doc pair
            if self.config.loss == 'logistic':
                loss_q_pos = tf.log1p(tf.reduce_sum(tf.exp(delta), axis=2))
            elif self.config.loss == 'hinge':
                loss_q_pos = tf.reduce_sum(tf.nn.relu(self.config.margin + delta), axis=2)
            elif self.config.loss == 'exp':
                loss_q_pos = tf.reduce_sum(tf.exp(delta), axis=2)
            else:
                raise NotImplementedError

            self.model_loss = tf.reduce_sum(loss_q_pos)

            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.model_loss)

        # precision
        with tf.name_scope("accuracy"):
            self.all_metric = tf.concat([self.pos_metric, self.neg_metric], axis=-1)
            self.y_pred = tf.argmax(self.all_metric, axis=-1)  # 取相似得分最大的一个为正例
            self.correct_pred = tf.equal(self.input_y, self.y_pred)
            self.acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


class LocalModelProcess(object):
    """
    本地模型处理类
    """

    def __init__(self, config_util, data_helper, local_model):
        self.config_util = config_util
        self.data_helper = data_helper
        self.local_model = local_model

        self.gpu_config = config_util.gpu_util()


    def load_local_train_data(self):
        """
        加载本地模型训练集
        :return:
        """
        print("Loading training data...")
        start_time = time.time()
        words, word_to_id = self.config_util.words, self.config_util.word_to_id
        # 载入训练集
        mention_train, entity_train, fea_train, url_train, y_train = self.data_helper.process_file(self.config_util.train_path, word_to_id,
                                                                               self.config_util.seq_length)

        entity_embedd_train = self.config_util.load_entity_embedd(url_train) # 根据url截取entity名，从而获得entity emb

        time_dif = self.data_helper.get_time_dif(start_time)
        print "Time usage:{0}".format(time_dif)
        print "mention_train:{0}, entity_train:{1}, fea_train:{2}, entity_embedd_train:{3}, y_train:{4}" \
            .format(len(mention_train), len(entity_train), len(fea_train), len(entity_embedd_train), len(y_train))
        return mention_train, entity_train, fea_train, entity_embedd_train, y_train

    def load_local_test_data(self):
        """
        加载本地模型测试集
        :return:
        """
        print("Loading test data...")
        start_time = time.time()
        words, word_to_id = self.config_util.words, self.config_util.word_to_id
        # 载入测试集
        mention_test, entity_test, fea_test, url_test, y_test = self.data_helper.process_file(self.config_util.test_path,
                                                                                        word_to_id,
                                                                                        self.config_util.seq_length)
        entity_embedd_test = self.config_util.load_entity_embedd(url_test)

        time_dif = self.data_helper.get_time_dif(start_time)
        print "Time usage:{0}".format(time_dif)
        print "mention_test:{0}, entity_test:{1}, fea_test:{2}, entity_embedd_test:{3}, y_test:{4}"\
            .format(len(mention_test), len(entity_test), len(fea_test), len(entity_embedd_test), len(y_test))

        return mention_test, entity_test, fea_test, entity_embedd_test, y_test

    def evaluate(self, sess, mention_, entity_, fea_, embedd_, y_):
        """
        评估在某一数据上的准确率和损失
        :param sess:
        :param mention_:
        :param entity_:
        :param embedd_:
        :param y_:
        :return:
        """
        data_len = len(mention_) / self.config_util.candidate_entity_num
        batch_eval = self.data_helper.batch_iter_local(mention_, entity_, fea_, embedd_, y_,
                                                       self.config_util.candidate_entity_num,
                                                       self.config_util.batch_size * 4)
        total_loss = 0.0
        total_acc = 0.0
        for mention_batch, pos_entity_batch, neg_entity_batch, pos_fea_batch, neg_fea_batch, \
            pos_embedd_batch, neg_emebdd_batch, y_batch in batch_eval:
            feed_dict = {
                self.local_model.input_mention: mention_batch,
                self.local_model.positive_entities: pos_entity_batch,
                self.local_model.negative_entities: neg_entity_batch,
                self.local_model.pos_entity_fea: pos_fea_batch,
                self.local_model.neg_entity_fea: neg_fea_batch,
                self.local_model.pos_entity_embedd: pos_embedd_batch,
                self.local_model.neg_entity_embedd: neg_emebdd_batch,
                self.local_model.input_y: y_batch,
                self.local_model.keep_prob: 1.0
            }
            batch_len = len(mention_batch)

            loss, acc = sess.run([self.local_model.model_loss, self.local_model.acc], feed_dict=feed_dict)
            total_loss += loss * batch_len
            total_acc += acc * batch_len

        return total_loss / data_len, total_acc / data_len

    def train_local_model(self):
        """
        训练本地模型
        :return:
        """

        # 配置本地模型保存路径
        saver = tf.train.Saver()
        if not os.path.exists(self.config_util.save_local_dir):
            os.makedirs(self.config_util.save_local_dir)

        # 预处理数据
        # mention id，entity id， feature， entity embedding，label
        mention_train, entity_train, fea_train, embedd_train, y_train = self.load_local_train_data()
        mention_test, entity_test, fea_test, embedd_test, y_test = self.load_local_test_data()

        # 创建session
        session = tf.Session(config=self.gpu_config)
        session.run(tf.global_variables_initializer())

        print('Training Local Model...')
        start_time = time.time()
        # 总批次
        total_batch = 0
        # 最佳验证集准确率
        best_acc_val = 0.0
        # 记录上一次提升批次
        last_improved = 0

        # early stopping的标志位
        early_stop_flag = False
        for epoch in range(self.config_util.local_num_epochs):
            print('Epoch:', epoch + 1)
            # 在batch_iter_local中yeild出来的是一个batch的mention, positive_list, neg_list组合
            batch_train = self.data_helper.batch_iter_local(mention_train, entity_train, fea_train, embedd_train, y_train,
                                                      self.config_util.candidate_entity_num,
                                                      self.config_util.batch_size, is_random=True)
            for mention_batch, pos_entity_batch, neg_entity_batch, pos_fea_batch, neg_fea_batch, \
                pos_embedd_batch, neg_emebdd_batch, y_batch in batch_train:

                feed_dict = {
                    self.local_model.input_mention: mention_batch,
                    self.local_model.positive_entities: pos_entity_batch,
                    self.local_model.negative_entities: neg_entity_batch,
                    self.local_model.pos_entity_fea: pos_fea_batch,
                    self.local_model.neg_entity_fea: neg_fea_batch,
                    self.local_model.pos_entity_embedd: pos_embedd_batch,
                    self.local_model.neg_entity_embedd: neg_emebdd_batch,
                    self.local_model.input_y: y_batch,
                    self.local_model.keep_prob: self.config_util.dropout_keep_prob
                }

                # 输出在训练集的loss
                if total_batch % self.config_util.print_per_batch == 0:
                    loss_train, acc_train = session.run([self.local_model.model_loss, self.local_model.acc], feed_dict=feed_dict)
                    time_dif = self.data_helper.get_time_dif(start_time)
                    msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                          + ' Time: {3}'
                    print(msg.format(total_batch, loss_train, acc_train, time_dif))

                    if total_batch % 50 == 0:
                        loss_val, acc_val = self.evaluate(session, mention_test, entity_test, fea_test, embedd_test, y_test)
                        # 保存最好结果
                        if acc_val > best_acc_val:
                            best_acc_val = acc_val
                            last_improved = total_batch
                            saver.save(sess=session, save_path=self.config_util.save_local_dir)
                            improved_str = '*'
                        else:
                            improved_str = ''

                        time_dif = self.data_helper.get_time_dif(start_time)
                        msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                              + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                        print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

                # 对loss进行优化
                session.run(self.local_model.optim, feed_dict=feed_dict)
                total_batch += 1

                # 正确率长期不提升，提前结束训练
                if total_batch - last_improved > self.config_util.local_require_improvement:
                    print("No optimization for a long time, auto-stopping...")
                    early_stop_flag = True
                    break

            # early stopping
            if early_stop_flag:
                break

        session.close()

    def test_local_model(self):
        """
        测试本地模型
        :return:
        """
        print "Test Local Model..."
        # 创建session
        session = tf.Session(config=self.gpu_config)
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # 读取训练好的模型
        saver.restore(sess=session, save_path=self.config_util.save_local_dir)

        mention_test, entity_test, fea_test, embedd_test, y_test = self.load_local_test_data()

        loss_val, acc_val = self.evaluate(session, mention_test, entity_test, fea_test, embedd_test, y_test)
        print "loss_val:{0}, acc_val:{1}".format(loss_val, acc_val)

        session.close()

    def save_local_representation(self, mention_, entity_, fea_, embedd_, y_, save_path):
        """
        保存本地模型中的中间表示向量
        :param mention_: 所有mention数据, numpy
        :param entity_: 所有entity数据, numpy
        :param fea_:
        :param embedd_:
        :param y_:
        :param save_path:
        :return:
        """
        print('Save Local Representation...')
        start_time = time.time()

        # 创建session
        session = tf.Session(config=self.gpu_config)
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # 读取训练好的模型
        saver.restore(sess=session, save_path=self.config_util.save_local_dir)

        local_represent_list = []
        batch_eval = self.data_helper.batch_iter_local(mention_, entity_, fea_, embedd_, y_,
                                                        self.config_util.candidate_entity_num,
                                                        self.config_util.batch_size * 4, is_rank=True)

        for mention_batch, pos_entity_batch, neg_entity_batch, pos_fea_batch, neg_fea_batch, \
            pos_embedd_batch, neg_emebdd_batch, y_batch in batch_eval:
            feed_dict = {
                self.local_model.input_mention: mention_batch,
                self.local_model.positive_entities: pos_entity_batch,
                self.local_model.negative_entities: neg_entity_batch,
                self.local_model.pos_entity_fea: pos_fea_batch,
                self.local_model.neg_entity_fea: neg_fea_batch,
                self.local_model.pos_entity_embedd: pos_embedd_batch,
                self.local_model.neg_entity_embedd: neg_emebdd_batch,
                self.local_model.keep_prob: 1.0
            }
            # mention_rep_shape=(B,H), entity_rep_shape=(B,P,H）
            mention_rep_batch, pos_entity_rep_batch, neg_entity_reps_batch = session.run(
                [self.local_model.mention_context_rep, self.local_model.pos_entity_reps,
                 self.local_model.neg_entity_reps], feed_dict=feed_dict)

            for index in range(len(mention_batch)):
                # shape=(H)
                mention_rep = mention_rep_batch[index]
                # shape=(P,H)
                pos_entity_reps = pos_entity_rep_batch[index]
                # shape=(N,H)
                neg_entity_reps = neg_entity_reps_batch[index]
                pos_index = y_batch[index]

                neg_index = 0
                for entity_index in range(config_util.candidate_entity_num):
                    if entity_index == pos_index:
                        mention_entity = np.vstack((mention_rep, pos_entity_reps[0]))
                    else:
                        mention_entity = np.vstack((mention_rep, neg_entity_reps[neg_index])) # [Vm(i), Ve(ik)]
                        neg_index += 1

                    local_represent_list.append(mention_entity)

        # 将中间表示存入文件中, shape=(batch, 2, local_representation_size)
        print "local_represent_list:{0}".format(len(local_represent_list))
        local_represent_np = np.array(local_represent_list)
        np.save(save_path, local_represent_np)

        time_dif = self.data_helper.get_time_dif(start_time)
        print("Save Representation Time usage:{}", time_dif)

        session.close()

    def rank_candidate(self, mention_, entity_, fea_, embedd_, y_,
                       source_data_path, rank_data_path, source_rep_path, rank_rep_path, is_random=False):
        """
        根据本地模型的预测结果对候选进行排序
        :param mention_: 所有mention数据, numpy
        :param entity_: 所有entity数据, numpy
        :param fea_:
        :param embedd_:
        :param y_:
        :param is_random: 是否以一定概率对预测的目标实体排序
        :return:
        """
        # 创建session
        session = tf.Session(config=self.gpu_config)
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # 读取训练好的模型
        saver.restore(sess=session, save_path=self.config_util.save_local_dir)

        batch_eval = self.data_helper.batch_iter_local(mention_, entity_, fea_, embedd_, y_,
                                                       self.config_util.candidate_entity_num,
                                                       self.config_util.batch_size * 4, is_rank=True)

        pred_index_list = []
        for mention_batch, pos_entity_batch, neg_entity_batch, pos_fea_batch, neg_fea_batch, \
            pos_embedd_batch, neg_emebdd_batch, y_batch in batch_eval:
            feed_dict = {
                self.local_model.input_mention: mention_batch,
                self.local_model.positive_entities: pos_entity_batch,
                self.local_model.negative_entities: neg_entity_batch,
                self.local_model.pos_entity_fea: pos_fea_batch,
                self.local_model.neg_entity_fea: neg_fea_batch,
                self.local_model.pos_entity_embedd: pos_embedd_batch,
                self.local_model.neg_entity_embedd: neg_emebdd_batch,
                self.local_model.input_y: np.zeros([self.config_util.batch_size * 4], dtype=np.int64),
                self.local_model.keep_prob: 1.0
            }

            # shape=(B,1)
            pred_batch = session.run(self.local_model.y_pred, feed_dict=feed_dict)
            tmp_pred_list = []
            # 记录本地模型预测的目标实体下标
            for index, pred_item in enumerate(pred_batch):
                if pred_item == 0:
                    tmp_pred_list.append(y_batch[index])
                else:
                    tmp_pred_list.append(pred_item)

            pred_index_list.extend(tmp_pred_list)

        print "pred_index_list:{}".format(len(pred_index_list))

        rank_candidate_list = []
        # 以一定概率将预测的目标实体放在首位
        if is_random:
            for count, pos_index in enumerate(pred_index_list):
                start_index = count * self.config_util.candidate_entity_num
                tmp_list = []

                tmp_choice = np.random.choice(2, p=[0.5, 0.5])
                if tmp_choice == 0:
                    # 将预测的目标实体放在首位
                    for candidate_index in range(self.config_util.candidate_entity_num):
                        if candidate_index == 0:
                            tmp_list.append(pos_index)
                        elif candidate_index != pos_index:
                            tmp_list.append(candidate_index)
                        elif candidate_index == pos_index:
                            tmp_list.append(0)
                else:
                    # 不排序
                    tmp_list.extend([candidate_index for candidate_index in range(self.config_util.candidate_entity_num)])

                rank_candidate_list.extend([start_index+ele for ele in tmp_list])

        else:
            for count, pos_index in enumerate(pred_index_list):
                start_index = count * self.config_util.candidate_entity_num
                tmp_list = []

                # 将预测的目标实体放在首位
                for candidate_index in range(self.config_util.candidate_entity_num):
                    if candidate_index == 0:
                        tmp_list.append(pos_index)
                    elif candidate_index != pos_index:
                        tmp_list.append(candidate_index)
                    elif candidate_index == pos_index:
                        tmp_list.append(0)

                rank_candidate_list.extend([start_index + ele for ele in tmp_list])

        print "rank_candidate_list:{}".format(len(rank_candidate_list))

        # 保存candidate entity排序后的候选实体文件
        train_entity_list = []
        with open(source_data_path, "r") as source_data_file:
            with open(rank_data_path, "w") as data_rank_file:
                for item in source_data_file:
                    item = item.strip().decode("utf-8")
                    train_entity_list.append(item)

                for rank_index in rank_candidate_list:
                    data_rank_file.write(train_entity_list[rank_index].encode("utf-8") + "\n")

        # 保存candidate entity排序后的本地表示文件
        local_rep_np = np.load(source_rep_path)
        local_rep_rank_list = []
        for rank_index in rank_candidate_list:
            local_rep_rank_list.append(local_rep_np[rank_index])

        print "local_rep_np:{0}, local_rep_rank_list:{1}".format(len(local_rep_np), len(local_rep_rank_list))
        np.save(rank_rep_path, np.array(local_rep_rank_list))


    def rank_mention(self, mention_, entity_, fea_, embedd_, y_,
                     source_data_path, rank_data_path, source_rep_path, rank_rep_path, group_path):
        """
        对文档中的mention按照所有mention数据本地模型相似度值排序
        :param mention_: , numpy
        :param entity_: 所有entity数据, numpy
        :param fea_:
        :param embedd_:
        :param y_:
        :param source_mention_path: 源mention数据文件路径
        :param rank_mention_path:  对mention数据按相似度排序后的文件路径
        :param source_mention_path: 源mention中间表示文件路径
        :param rank_mention_path:  对mention中间表示按相似度排序后的文件路径
        :param group_path:  group文件路径
        :return:
        """
        print('Rank Mention...')
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

        print "group_mention_num:{0}".format(sum([sum(item) for item in all_doc_list]))

        # 创建session
        session = tf.Session(config=self.gpu_config)
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # 读取训练好的本地模型
        saver.restore(sess=session, save_path=self.config_util.save_local_dir)

        batch_eval = self.data_helper.batch_iter_local(mention_, entity_, fea_, embedd_, y_,
                                                       self.config_util.candidate_entity_num,
                                                       self.config_util.batch_size * 4, is_rank=True)
        all_sim_list = []
        for mention_batch, pos_entity_batch, neg_entity_batch, pos_fea_batch, neg_fea_batch, \
            pos_embedd_batch, neg_emebdd_batch, y_batch in batch_eval:
            feed_dict = {
                self.local_model.input_mention: mention_batch,
                self.local_model.positive_entities: pos_entity_batch,
                self.local_model.negative_entities: neg_entity_batch,
                self.local_model.pos_entity_fea: pos_fea_batch,
                self.local_model.neg_entity_fea: neg_fea_batch,
                self.local_model.pos_entity_embedd: pos_embedd_batch,
                self.local_model.neg_entity_embedd: neg_emebdd_batch,
                self.local_model.keep_prob: 1.0
            }
            # shape=(B,P), shape=(B,N)
            pos_sim, neg_sim = session.run([self.local_model.pos_metric, self.local_model.neg_metric], feed_dict=feed_dict)

            # mention的最大相似度值,shape=(B,1)
            max_sim = np.max(np.concatenate([pos_sim, neg_sim], axis=-1), axis=-1)
            all_sim_list.extend([item for item in max_sim])

        max_sim_list = []
        # 排序后的index列表
        rank_entity_list = []
        entity_start_index = 0
        doc_index = 0
        doc_num = len(all_doc_list)
        doc_mention_num = len(all_doc_list[doc_index])
        for max_sim in all_sim_list:
            max_sim_list.append(max_sim)

            # 对一个doc内的mention进行排序
            if len(max_sim_list) == doc_mention_num:
                doc_group_num = int((doc_mention_num - 1) / self.config_util.rank_mention_num) + 1
                # 对一个group内的mention进行排序
                for doc_group_index in range(doc_group_num):
                    start_id = doc_group_index * self.config_util.rank_mention_num
                    end_id = min((doc_group_index + 1) * self.config_util.rank_mention_num, doc_mention_num)
                    tmp_sim_list = max_sim_list[start_id:end_id]

                    # 按照每个mention的最大相似度排序
                    tmp_dict = {}
                    for index, val in enumerate(tmp_sim_list):
                        tmp_dict[index] = val
                    rank_mention_index_list = [item[0] for item in sorted(tmp_dict.items(), key=lambda x: x[1], reverse=True)]

                    for rank_mention_index in rank_mention_index_list:
                        rank_entity_list.extend([entity_start_index + entity_index for entity_index in
                                                 range(self.config_util.candidate_entity_num * rank_mention_index,
                                                       self.config_util.candidate_entity_num * (rank_mention_index + 1))])

                    entity_start_index += (end_id - start_id) * self.config_util.candidate_entity_num

                max_sim_list = []
                doc_index += 1
                if doc_index < doc_num:
                    doc_mention_num = len(all_doc_list[doc_index])

        print "rank_entity_list:{0}, set(rank_entity_list):{1}, mention_num:{2}"\
            .format(len(rank_entity_list), len(set(rank_entity_list)), len(mention_))

        # 保存mention排序后的候选实体文件
        train_entity_list = []
        with open(source_data_path, "r") as source_data_file:
            with open(rank_data_path, "w") as data_rank_file:
                for item in source_data_file:
                    item = item.strip().decode("utf-8")
                    train_entity_list.append(item)

                for rank_index in rank_entity_list:
                    data_rank_file.write(train_entity_list[rank_index].encode("utf-8") + "\n")

        # 保存排序后的本地表示文件
        local_rep_np = np.load(source_rep_path)
        local_rep_rank_list = []
        for rank_index in rank_entity_list:
            local_rep_rank_list.append(local_rep_np[rank_index])

        print "local_rep_np:{0}, local_rep_rank_list:{1}".format(len(local_rep_np), len(local_rep_rank_list))
        np.save(rank_rep_path, np.array(local_rep_rank_list))


    def controller(self):
        """

        :return:
        """
        # 训练本地模型
        local_model_process.train_local_model()

        mention_train, entity_train, fea_train, embedd_train, y_train = self.load_local_train_data() # 前面的读取都是一个mention一个entity，直到batch_iter_local时候才会转化成一个mention一组entity
        # 保存训练文件中间表示
        local_model_process.save_local_representation(mention_train, entity_train, fea_train, embedd_train, y_train,
                                                      self.config_util.local_rep_train_path)

        # 训练文件候选实体相似度排序
        self.rank_candidate(mention_train, entity_train, fea_train, embedd_train, y_train,
                            self.config_util.train_path, self.config_util.train_candidate_rank_path,
                            self.config_util.local_rep_train_path,
                            self.config_util.local_rep_candidate_rank_train_path)

        # 训练文件mention相似度排序
        self.rank_mention(mention_train, entity_train, fea_train, embedd_train, y_train,
                          self.config_util.train_candidate_rank_path, self.config_util.train_mention_rank_path,
                          self.config_util.local_rep_candidate_rank_train_path,
                          self.config_util.local_rep_mention_rank_train_path,
                          self.config_util.group_path)

        # # 测试本地模型
        local_model_process.test_local_model()

        mention_test, entity_test, fea_test, embedd_test, y_test = self.load_local_test_data()
        # 保存测试文件中间表示
        local_model_process.save_local_representation(mention_test, entity_test, fea_test, embedd_test, y_test,
                                                      self.config_util.local_rep_test_path)
        # 测试文件候选实体相似度排序
        self.rank_candidate(mention_test, entity_test, fea_test, embedd_test, y_test,
                            self.config_util.test_path, self.config_util.test_candidate_rank_path,
                            self.config_util.local_rep_test_path, self.config_util.local_rep_candidate_rank_test_path)
        # 测试文件mention相似度排序
        self.rank_mention(mention_test, entity_test, fea_test, embedd_test, y_test,
                          self.config_util.test_candidate_rank_path, self.config_util.test_mention_rank_path,
                          self.config_util.local_rep_candidate_rank_test_path,
                          self.config_util.local_rep_mention_rank_test_path,
                          self.config_util.test_group_path)



if __name__ == "__main__":
    data_helper = DataUtil()
    config_util = LocalConfigUtil(data_helper)
    local_model = LocalModel(config_util)
    local_model_process = LocalModelProcess(config_util, data_helper, local_model)
    local_model_process.controller()

