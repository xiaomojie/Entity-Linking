# encoding: utf-8

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import time
from global_config import *
from data_util import *

class GlobalModel(object):

    def __init__(self, config):
        self.config = config

        # mention上下文表示, shape=(B,S,H)
        self.input_mention_context = tf.placeholder(tf.float32, [None, None, self.config.local_rep_dim],
                                                    name="input_mention_context")
        # 候选实体表示
        self.input_candidate_entity_local = tf.placeholder(tf.float32, [None, None, self.config.local_rep_dim],
                                                     name="input_candidate_entity_local")

        # 候选实体向量
        self.input_candidate_entity_embedding = tf.placeholder(tf.float32, [None, None, self.config.entity_embedding_dim],
                                               name="input_candidate_entity_embedding")

        # 候选实体先验概率
        self.input_candidate_entity_priori = tf.placeholder(tf.float32, [None, None, self.config.priori_dim],
                                                            name="input_candidate_entity_priori")

        # 输出类别
        self.input_y = tf.placeholder(tf.int32, [None, None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")


        self.rnn_model()


    def rnn_model(self):
        """
        构造全局表示网络
        :return:
        """
        # lstm核
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size, state_is_tuple=True)

        # gru核
        def gru_cell():
            return tf.contrib.rnn.GRUCell(self.config.hidden_size)

        # 在rnn核后加dropout层
        def dropout():
            if (self.config.rnn == "lstm"):
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

        # 将mention表示与候选实体表示拼接
        entity_concat = tf.concat([self.input_candidate_entity_local,
                                   self.input_candidate_entity_embedding], axis=-1)

        new_input = tf.concat([self.input_mention_context, entity_concat], axis=-1)  # [m context, e con, e embedding ]

        _output = rnn_output(new_input)
        self.rnn_output = _output
        mlp_output = _output
        # lstm 隐藏层的输出再经过两层的mlp
        for l_size in ([32, 16][:self.config.mlp_metric_layer]):
            mlp_output = slim.fully_connected(mlp_output, l_size)

        # 全连接转化成2维输出
        self.output_logits = slim.fully_connected(mlp_output, self.config.num_classes, activation_fn=None)

        # 预测类别
        self.y_pred_cls = tf.argmax(tf.nn.softmax(self.output_logits), -1)
        # 损失函数，交叉熵
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.output_logits, labels=self.input_y)
        self.loss = tf.reduce_mean(cross_entropy)
        # 优化器
        self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
        # acc
        correct_pred = tf.equal(tf.argmax(self.input_y, -1), self.y_pred_cls)
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


class GlobalModelProcess(object):
    """
    全局模型处理类
    """

    def __init__(self, config_util, data_helper, global_model):
        """

        :param config_util:
        :param data_helper:
        :param global_model:
        """
        self.config_util = config_util
        self.data_helper = data_helper
        self.global_model = global_model

        self.gpu_config = config_util.gpu_util()

    def load_global_train_data(self):
        """
        加载全局模型的训练数据
        :return:
        """
        print("Loading training data...")
        start_time = time.time()
        # 载入训练集
        mention_train, entity_train, feature_train, entity_url_train, y_train = self.data_helper.process_global_data(
            self.config_util.train_rank_path,
            self.config_util.train_rep_rank_mention,
            self.config_util.train_group_path,
            self.config_util.seq_mention_num,
            self.config_util.candidate_entity_num)

        # 根据url查找entity embedding
        entity_embedd_train = self.config_util.load_entity_embedd(entity_url_train)

        time_dif = self.data_helper.get_time_dif(start_time)
        print("Time usage:{0}".format(time_dif))
        return mention_train, entity_train, feature_train, entity_embedd_train, y_train

    def load_global_test_data(self):
        """
        加载全局模型的测试数据
        :return:
        """
        print("Loading test data...")
        start_time = time.time()
        # 载入训练集
        mention_test, entity_test, feature_test, entity_url_test, y_test = self.data_helper.process_global_data(
            self.config_util.test_rank_path,
            self.config_util.test_rep_rank_mention,
            self.config_util.test_group_path,
            self.config_util.seq_mention_num,
            self.config_util.candidate_entity_num)

        # 根据url查找entity embedding
        entity_embedd_test = self.config_util.load_entity_embedd(entity_url_test)

        time_dif = self.data_helper.get_time_dif(start_time)
        print("Time usage:{0}".format(time_dif))

        return mention_test, entity_test, feature_test, entity_embedd_test, y_test


    def evaluate(self, sess, mention_, entity_, fea_, embedd_, y_):
        """
        评估在某一数据上的准确率和损失
        :param sess:
        :param mention_:
        :param entity_:
        :param fea_:
        :param embedd_:
        :param y_:
        :return:
        """
        data_len = len(mention_) / (self.config_util.seq_mention_num * self.config_util.candidate_entity_num)
        batch_eval = self.data_helper.batch_iter_global_train(mention_, entity_, fea_, embedd_, y_,
                                                               self.config_util.seq_mention_num,
                                                               self.config_util.candidate_entity_num,
                                                               self.config_util.batch_size)
        total_loss = 0.0
        total_acc = 0.0
        for mention_batch, entity_batch, fea_batch, embedd_batch, y_batch in batch_eval:
            batch_len = len(mention_batch)
            feed_dict = {
                self.global_model.input_mention_context: mention_batch,
                self.global_model.input_candidate_entity_local: entity_batch,
                self.global_model.input_candidate_entity_embedding: embedd_batch,
                self.global_model.input_candidate_entity_priori: fea_batch,
                self.global_model.input_y: y_batch,
                self.global_model.keep_prob: 1.0
            }
            loss, acc = sess.run([self.global_model.loss, self.global_model.acc], feed_dict=feed_dict)
            total_loss += loss * batch_len
            total_acc += acc * batch_len

        return total_loss / data_len, total_acc / data_len


    def train_global_model(self):
        """
        训练全局模型
        :return:
        """

        # 配置全局模型保存路径
        saver = tf.train.Saver()
        if not os.path.exists(self.config_util.save_global_dir):
            os.makedirs(self.config_util.save_global_dir)

        # 预处理数据
        mention_train, entity_train, feature_train, entity_embedd_train, y_train = self.load_global_train_data()
        mention_test, entity_test, feature_test, entity_embedd_test, y_test = self.load_global_test_data()

        # 创建session
        session = tf.Session(config=self.gpu_config)
        session.run(tf.global_variables_initializer())

        print('Training Global Model...')
        start_time = time.time()
        # 总批次
        total_batch = 0
        # 最佳验证集准确率
        best_acc_val = 0.0
        # 记录上一次提升批次
        last_improved = 0

        # early stopping的标志位
        early_stop_flag = False
        for epoch in range(self.config_util.global_num_epochs):
            print('Epoch:', epoch + 1)
            batch_train = self.data_helper.batch_iter_global_train(mention_train, entity_train,
                                                                   feature_train, entity_embedd_train, y_train,
                                                                   self.config_util.seq_mention_num,
                                                                   self.config_util.candidate_entity_num,
                                                                   self.config_util.batch_size)
            for mention_batch, entity_batch, fea_batch, embedd_batch, y_batch in batch_train:
                feed_dict = {
                    self.global_model.input_mention_context: mention_batch,
                    self.global_model.input_candidate_entity_local: entity_batch,
                    self.global_model.input_candidate_entity_embedding: embedd_batch,
                    self.global_model.input_candidate_entity_priori: fea_batch,
                    self.global_model.input_y: y_batch,
                    self.global_model.keep_prob: self.config_util.dropout_keep_prob
                }

                # 输出在训练集上的性能
                if total_batch % self.config_util.print_per_batch == 0:
                    feed_dict[self.global_model.keep_prob] = 1.0
                    loss_train, acc_train = session.run([self.global_model.loss, self.global_model.acc], feed_dict=feed_dict)
                    time_dif = self.data_helper.get_time_dif(start_time)
                    msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                          + ' Time: {3}'
                    print(msg.format(total_batch, loss_train, acc_train, time_dif))

                    if total_batch % 25 == 0:
                        loss_val, acc_val = self.evaluate(session, mention_test, entity_test,
                                                          feature_test, entity_embedd_test, y_test)

                        # 保存最好结果
                        if acc_val > best_acc_val:
                            best_acc_val = acc_val
                            last_improved = total_batch
                            saver.save(sess=session, save_path=self.config_util.save_global_dir)
                            improved_str = '*'
                        else:
                            improved_str = ''

                        time_dif = self.data_helper.get_time_dif(start_time)
                        msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                              + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                        print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

                # 对loss进行优化
                session.run(self.global_model.optim, feed_dict=feed_dict)
                total_batch += 1

                # 验证集正确率长期不提升，提前结束训练
                if total_batch - last_improved > self.config_util.local_require_improvement:
                    print("No optimization for a long time, auto-stopping...")
                    early_stop_flag = True
                    break

            # early stopping
            if early_stop_flag:
                break

        session.close()

    def controller(self):
        """

        :return:
        """
        # 训练全局模型
        self.train_global_model()


if __name__ == "__main__":
    global_config_util = GlobalConfigUtil()
    data_helper = DataUtil()
    global_model = GlobalModel(global_config_util)
    global_model_process = GlobalModelProcess(global_config_util, data_helper, global_model)
    global_model_process.controller()