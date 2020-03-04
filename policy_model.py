# encoding: utf-8

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from global_config import *
from data_util import *
import os
from global_model import *

class Agent():
    """
    强化学习中agent类，与环境进行交互，此处即为一个神经网络
    """

    def __init__(self, config):
        self.config = config

        # mention上下文表示, shape=(B, S, H)
        self.input_mention_context = tf.placeholder(tf.float32, [None, None, self.config.local_rep_dim],
                                                    name="input_mention_context")
        # 候选实体本地表示
        self.input_candidate_entity = tf.placeholder(tf.float32, [None, None, self.config.candidate_entity_num, self.config.local_rep_dim],
                                                     name="input_candidate_entity")
        # 候选实体向量
        self.entity_embedding = tf.placeholder(tf.float32, [None, None, self.config.candidate_entity_num, self.config.entity_embedding_dim],
                                               name="entity_embedding")

        # 候选实体先验概率
        self.input_candidate_entity_priori = tf.placeholder(tf.float32, [None, None, self.config.candidate_entity_num, self.config.priori_dim],
                                                            name="input_entity_pv")

        # 上一次决策后的状态
        self.last_state = tf.placeholder(tf.float32, [None, None, self.config.hidden_size], name="last_state")

        # action和reward
        self.action_holder = tf.placeholder(shape=[None, self.config.seq_mention_num], dtype=tf.int32)
        self.reward_holder = tf.placeholder(shape=[None, self.config.seq_mention_num], dtype=tf.float32)  # reward的长度为什么也是4？不是一个seq玩了之后一次reward吗

        self.policy_model()


    def policy_model(self):
        """
        构造policy网络，选择action
        :return:
        """
        # shape=(B, S, 1, H)
        mention_norm = tf.expand_dims(self.input_mention_context, 2)
        entity_local_norm = self.input_candidate_entity
        entity_embedding_norm = self.entity_embedding
        entity_priori_norm = self.input_candidate_entity_priori
        last_state_norm = tf.expand_dims(self.last_state, 2)

        # 将候选实体与先验特征相拼接
        entity_fea_input = tf.concat([entity_local_norm, entity_embedding_norm, entity_priori_norm], axis=-1)
        # 将mention与候选实体进行拼接, shape=(B,S,C,H)，复制成候选实体的个数
        mention_entity = tf.concat([tf.tile(mention_norm, [1, 1, self.config.candidate_entity_num, 1]), entity_fea_input], axis=-1)
        # 将当前输入和上一次状态进行拼接
        new_input = tf.concat([tf.tile(last_state_norm, [1, 1, self.config.candidate_entity_num, 1]), mention_entity], axis=-1)

        mlp_output = new_input
        # 4层的mlp加上一个一维的输出score
        for l_size in ([128, 64, 32, 16][:self.config.policy_mlp_layer] + [1]):
            mlp_output = slim.fully_connected(mlp_output, l_size, activation_fn=tf.nn.softplus)

        # shape=(B,S,C)
        self.action_output = tf.nn.softmax((tf.squeeze(mlp_output, squeeze_dims=[-1])))

        # 根据action选择当前候选实体
        self.chosen_action = tf.argmax(self.action_output, -1)

        # 获取文档中每次选择action的下标
        self.indexes = tf.range(0, tf.shape(self.action_output)[0] * tf.shape(self.action_output)[1]) * tf.shape(self.action_output)[2]\
                       + tf.reshape(self.action_holder, [-1])
        # 根据indexs获取output中相应的值
        self.responsible_outputs = tf.gather(tf.reshape(self.action_output, [-1]), self.indexes)
        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * tf.reshape(self.reward_holder, [-1])) #action ouput * action reward

        # 优化器
        self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        # tvars = tf.trainable_variables()
        # self.gradients = tf.gradients(self.loss, tvars)
        #
        # self.gradient_holders = []
        # for idx, var in enumerate(tvars):
        #     placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
        #     self.gradient_holders.append(placeholder)
        #
        # optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        # self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))


class AgentProcess(object):
    """
    agent处理类
    """

    def __init__(self, config_util, data_helper):
        self.config_util = config_util
        self.data_helper = data_helper

        self.gpu_config = config_util.gpu_util()


    def load_policy_train_data(self):
        """
        加载训练数据
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
        print "Time usage:{0}".format(time_dif)
        return mention_train, entity_train, feature_train, entity_embedd_train, y_train

    def load_policy_test_data(self):
        """
        加载测试数据
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
        print "Time usage:{0}".format(time_dif)

        return mention_test, entity_test, feature_test, entity_embedd_test, y_test


    def discount_rewards(self, reward_np):
        """
        计算action_value
        :param reward_np: shape=(B,S)
        :return:
        """
        dicount_reward_list = []
        for reward_seq in reward_np:
            discounted_reward = np.zeros_like(reward_seq)
            penalty = 0
            add_reward = 0
            for time_step in reversed(xrange(0, len(reward_seq))):
                if reward_seq[time_step] == 0:
                    penalty += -1
                    discounted_reward[time_step] = penalty
                elif reward_seq[time_step] == 1:
                    add_reward += 1
                    discounted_reward[time_step] = add_reward
            dicount_reward_list.append(discounted_reward)

        return np.array(dicount_reward_list)

    def evaluate(self, policy_sess, global_sess, my_agent, global_model, mention_, entity_, fea_, embedd_, y_):
        """
        评估在某一数据上的准确率和损失
        :param policy_sess:
        :param global_sess:
        :param my_agent:
        :param global_model:
        :param mention_:
        :param entity_:
        :param fea_:
        :param embedd_:
        :param y_:
        :return:
        """
        data_len = len(mention_) / (self.config_util.seq_mention_num * self.config_util.candidate_entity_num)
        total_loss = 0.0
        total_acc = 0.0

        batch_train = self.data_helper.batch_iter_policy_train(mention_, entity_, fea_, embedd_, y_,
                                                               self.config_util.seq_mention_num,
                                                               self.config_util.candidate_entity_num,
                                                               self.config_util.batch_size)

        # mention_batch:shape=(B,S,H), (entity_batch,y_batch,fea_batch):shape=(B,S,C,H)
        for mention_batch, entity_batch, fea_batch, embedd_batch, y_batch in batch_train:
            # global模型中上一时刻的输出
            global_last_state = np.zeros([len(mention_batch), 1, self.config_util.hidden_size])

            seq_mention_list = []
            seq_select_entity_list = []
            seq_select_fea_list = []
            seq_select_embedd_list = []
            seq_last_state_list = []
            policy_action_list = []
            policy_reward_list = []

            for mention_index in range(self.config_util.seq_mention_num):
                current_mention_batch = mention_batch[:, mention_index, :]
                current_mention_batch = np.expand_dims(current_mention_batch, axis=1)
                cuurent_entity_batch = entity_batch[:, mention_index, :, :]
                cuurent_entity_batch = np.expand_dims(cuurent_entity_batch, axis=1)
                current_entity_fea_batch = fea_batch[:, mention_index, :, :]
                current_entity_fea_batch = np.expand_dims(current_entity_fea_batch, axis=1)
                current_entity_embedd_batch = embedd_batch[:, mention_index, :, :]
                current_entity_embedd_batch = np.expand_dims(current_entity_embedd_batch, axis=1)

                policy_feed_dict = {
                    my_agent.input_mention_context: current_mention_batch,
                    my_agent.input_candidate_entity: cuurent_entity_batch,
                    my_agent.input_candidate_entity_priori: current_entity_fea_batch,
                    my_agent.entity_embedding: current_entity_embedd_batch,
                    my_agent.last_state: global_last_state
                }

                # shape=(S,B,H)
                seq_last_state_list.append(np.squeeze(global_last_state, axis=1))

                # 选择概率最大的action
                chosen_actions = policy_sess.run(my_agent.chosen_action, feed_dict=policy_feed_dict)
                chosen_actions = np.squeeze(chosen_actions, axis=1)
                # print chosen_actions

                current_action_batch = [action for action in chosen_actions]

                # 获取即时reward
                current_y = y_batch[:, mention_index, :, :]
                current_label_batch = [np.argmax(current_y[batch_index, action_index, :], axis=-1)
                                       for batch_index, action_index in enumerate(current_action_batch)]
                current_reward_batch = []
                for label in current_label_batch:
                    # 正例
                    if label == 0:
                        current_reward_batch.append(1)
                    # 负例
                    elif label == 1:
                        current_reward_batch.append(0)

                # shape=(S,B)
                policy_action_list.append(current_action_batch)
                policy_reward_list.append(current_reward_batch)

                # 构造global模型的输入数据
                seq_mention_list.append(mention_batch[:, mention_index, :])
                global_mention_input = np.stack(seq_mention_list, axis=1)

                candidate_entity_batch = entity_batch[:, mention_index, :, :]
                entity_fea_batch = fea_batch[:, mention_index, :, :]
                entity_embedd_batch = embedd_batch[:, mention_index, :, :]
                select_entity_batch = [candidate_entity_batch[batch_index, select_index, :]
                                       for batch_index, select_index in enumerate(current_action_batch)]
                select_entity_fea_batch = [entity_fea_batch[batch_index, select_index, :]
                                           for batch_index, select_index in enumerate(current_action_batch)]
                select_entity_embedd_batch = [entity_embedd_batch[batch_index, select_index, :]
                                           for batch_index, select_index in enumerate(current_action_batch)]

                seq_select_entity_list.append(select_entity_batch)
                seq_select_fea_list.append(select_entity_fea_batch)
                seq_select_embedd_list.append(select_entity_embedd_batch)

                global_entity_input = np.stack(seq_select_entity_list, axis=1)
                global_fea_input = np.stack(seq_select_fea_list, axis=1)
                global_embedd_input = np.stack(seq_select_embedd_list, axis=1)

                global_feed_dict = {
                    global_model.input_mention_context: global_mention_input,
                    global_model.input_candidate_entity_local: global_entity_input,
                    global_model.input_candidate_entity_embedding: global_embedd_input,
                    global_model.input_candidate_entity_priori: global_fea_input,
                    global_model.keep_prob: 1.0
                }

                # 获取global模型的输出
                global_cell_output = global_sess.run(global_model.rnn_output, feed_dict=global_feed_dict)
                global_last_state = global_cell_output[:, -1, :]
                global_last_state = np.expand_dims(global_last_state, axis=1)

            # 周期结束
            # shape=(B,S)
            policy_action_np = np.stack(policy_action_list, axis=1)
            # 计算action_value
            policy_reward_np = np.stack(policy_reward_list, axis=1)
            policy_action_value = self.discount_rewards(policy_reward_np)

            seq_feed_dict = {
                my_agent.action_holder: policy_action_np,
                my_agent.reward_holder: policy_action_value,
                my_agent.input_mention_context: mention_batch,
                my_agent.input_candidate_entity: entity_batch,
                my_agent.input_candidate_entity_priori: fea_batch,
                my_agent.entity_embedding: embedd_batch,
                my_agent.last_state: np.stack(seq_last_state_list, axis=1)
            }

            batch_len = len(mention_batch)
            loss = policy_sess.run(my_agent.loss, feed_dict=seq_feed_dict)
            acc = float(sum([sum(seq_reward) for seq_reward in policy_reward_np])) / policy_reward_np.size
            total_loss += loss * batch_len
            total_acc += acc * batch_len

        return total_loss / data_len, total_acc / data_len

    def train_agent(self):
        """
        训练agent
        :return:
        """
        # 创建全局模型的graph
        global_graph = tf.Graph()
        global_session = tf.Session(graph=global_graph, config=self.gpu_config)
        # 加载全局模型
        with global_session.as_default():
            with global_graph.as_default():
                global_model = GlobalModel(self.config_util)
                global_session.run(tf.global_variables_initializer())
                global_saver = tf.train.Saver()
                global_saver.restore(sess=global_session, save_path=self.config_util.save_global_dir)

        # 创建policy模型的graph
        policy_graph = tf.Graph()
        policy_session = tf.Session(graph=policy_graph, config=self.gpu_config)
        # 训练policy模型
        with policy_session.as_default():
            with policy_graph.as_default():
                # 初始化agent,即初始化policy网络
                my_agent = Agent(self.config_util)
                policy_saver = tf.train.Saver()

                policy_session.run(tf.global_variables_initializer())

                # 加载训练数据
                mention_train, entity_train, fea_train, entity_embedd_train, y_train = self.load_policy_train_data()
                # 加载测试数据
                mention_test, entity_test, fea_test, entity_embedd_test, y_test = self.load_policy_test_data()

                print('Training Policy Model...')
                start_time = time.time()
                # 总批次
                total_batch = 0
                # 最佳验证集准确率
                best_acc_val = 0.0
                # 记录上一次提升批次
                last_improved = 0

                # early stopping的标志位
                early_stop_flag = False
                for epoch in range(self.config_util.policy_num_epochs):
                    print('Epoch:', epoch + 1)
                    batch_train = self.data_helper.batch_iter_policy_train(mention_train, entity_train,
                                                                           fea_train, entity_embedd_train, y_train,
                                                                           self.config_util.seq_mention_num,
                                                                           self.config_util.candidate_entity_num,
                                                                           self.config_util.batch_size,
                                                                           is_random=True)

                    # mention_batch:shape=(B,S,H), (entity_batch,y_batch,fea_batch):shape=(B,S,C,H)
                    for mention_batch, entity_batch, fea_batch, entity_embedd_batch, y_batch in batch_train:

                        # global模型中上一时刻的输出
                        global_last_state = np.zeros([len(mention_batch), 1, self.config_util.hidden_size])
                        # 采样得到的序列结果
                        seq_mention_list = []
                        seq_select_entity_list = []
                        seq_select_fea_list = []
                        seq_select_embedd_list = []
                        seq_last_state_list = []
                        policy_action_list = []
                        policy_reward_list = []

                        # 采样一个周期
                        for mention_index in range(self.config_util.seq_mention_num):
                            current_mention_batch = mention_batch[:, mention_index, :]
                            current_mention_batch = np.expand_dims(current_mention_batch, axis=1)
                            cuurent_entity_batch = entity_batch[:, mention_index, :, :]
                            cuurent_entity_batch = np.expand_dims(cuurent_entity_batch, axis=1)
                            current_entity_fea_batch = fea_batch[:, mention_index, :, :]
                            current_entity_fea_batch = np.expand_dims(current_entity_fea_batch, axis=1)
                            current_entity_embedd_batch = entity_embedd_batch[:, mention_index, :, :]
                            current_entity_embedd_batch = np.expand_dims(current_entity_embedd_batch, axis=1)

                            policy_feed_dict = {
                                my_agent.input_mention_context: current_mention_batch,
                                my_agent.input_candidate_entity: cuurent_entity_batch,
                                my_agent.input_candidate_entity_priori: current_entity_fea_batch,
                                my_agent.entity_embedding: current_entity_embedd_batch,
                                my_agent.last_state: global_last_state
                            }

                            # shape=(S,B,H)
                            seq_last_state_list.append(np.squeeze(global_last_state, axis=1))

                            # 通过output以一定概率形式选择action
                            action_dist = policy_session.run(my_agent.action_output, feed_dict=policy_feed_dict)
                            action_dist = np.squeeze(action_dist, axis=1)
                            current_action_batch = []
                            for action_prob in action_dist:
                                temp_action_prob = [prob/sum(action_prob) for prob in action_prob] # 每一个action的概率

                                # 以0.5的概率选择最大的action
                                # max_action = temp_action_prob.index(max(temp_action_prob))
                                # tmp_sum = sum([ele for index, ele in enumerate(temp_action_prob) if index != max_action])
                                #
                                # if tmp_sum != 0:
                                #     new_action_prob = [ele / tmp_sum * 0.5 for ele in temp_action_prob]
                                #     new_action_prob[max_action] = 0.5
                                # else:
                                #     new_action_prob = temp_action_prob

                                # 直接按照生成的概率随机选择
                                new_action_prob = temp_action_prob

                                action = np.random.choice(len(new_action_prob), p=new_action_prob) # 被选中的action的index
                                current_action_batch.append(action)

                            # 选择概率最大的action
                            # chosen_actions = policy_session.run(my_agent.chosen_action, feed_dict=policy_feed_dict)
                            # chosen_actions = np.squeeze(chosen_actions, axis=1)
                            # current_action_batch = []
                            # for action in chosen_actions:
                            #     current_action_batch.append(action)

                            # 获取即时reward
                            current_y = y_batch[:, mention_index, :, :]
                            current_label_batch = [np.argmax(current_y[batch_index, action_index, :], axis=-1)
                                            for batch_index, action_index in enumerate(current_action_batch)]
                            current_reward_batch = []  # 存的是p(a)
                            for label in current_label_batch:
                                # 正例
                                if label == 0:
                                    current_reward_batch.append(1)
                                # 负例
                                elif label == 1:
                                    current_reward_batch.append(0)
                            # shape=(S,B)
                            policy_action_list.append(current_action_batch) # action的集合
                            policy_reward_list.append(current_reward_batch) # 存的是p(a)的集合

                            # 构造global模型的输入数据
                            # shape=(S, B, H)
                            seq_mention_list.append(mention_batch[:, mention_index, :])
                            # shape=(B, S, H)
                            global_mention_input = np.stack(seq_mention_list, axis=1)

                            candidate_entity_batch = entity_batch[:, mention_index, :, :]
                            entity_fea_batch = fea_batch[:, mention_index, :, :]
                            embedd_batch = entity_embedd_batch[:, mention_index, :, :]
                            select_entity_batch = [candidate_entity_batch[batch_index, select_index, :]
                                                   for batch_index, select_index in enumerate(current_action_batch)]
                            select_entity_fea_batch = [entity_fea_batch[batch_index, select_index, :]
                                                   for batch_index, select_index in enumerate(current_action_batch)]
                            select_entity_embedd_batch = [embedd_batch[batch_index, select_index, :]
                                                       for batch_index, select_index in enumerate(current_action_batch)]
                            # shape=(S, B, H)
                            seq_select_entity_list.append(select_entity_batch)
                            seq_select_fea_list.append(select_entity_fea_batch)
                            seq_select_embedd_list.append(select_entity_embedd_batch)

                            # shape=(B,S,H)
                            global_entity_input = np.stack(seq_select_entity_list, axis=1)
                            global_fea_input = np.stack(seq_select_fea_list, axis=1)
                            global_embedd_input = np.stack(seq_select_embedd_list, axis=1)

                            global_feed_dict = {
                                global_model.input_mention_context: global_mention_input,
                                global_model.input_candidate_entity_local: global_entity_input,
                                global_model.input_candidate_entity_embedding: global_embedd_input,
                                global_model.input_candidate_entity_priori: global_fea_input,
                                global_model.keep_prob: 1.0
                            }
                            # 获取global模型的输出
                            global_cell_output = global_session.run(global_model.rnn_output, feed_dict=global_feed_dict)
                            global_last_state = global_cell_output[:, -1, :]
                            global_last_state = np.expand_dims(global_last_state, axis=1)

                        # 周期结束,更新网络
                        # shape=(B,S)
                        policy_action_np = np.stack(policy_action_list, axis=1)
                        # print policy_action_np

                        # 计算action_value
                        policy_reward_np = np.stack(policy_reward_list, axis=1)
                        policy_action_value = self.discount_rewards(policy_reward_np)

                        seq_feed_dict = {
                            my_agent.action_holder: policy_action_np,
                            my_agent.reward_holder: policy_action_value,
                            my_agent.input_mention_context: mention_batch,
                            my_agent.input_candidate_entity: entity_batch,
                            my_agent.input_candidate_entity_priori: fea_batch,
                            my_agent.entity_embedding: entity_embedd_batch,
                            my_agent.last_state: np.stack(seq_last_state_list, axis=1)
                        }

                        # 输出在测试集上的性能
                        if total_batch % self.config_util.print_per_batch == 0:
                            loss_train = policy_session.run(my_agent.loss, feed_dict=seq_feed_dict)
                            acc_train = float(sum([sum(seq_reward) for seq_reward in policy_reward_np])) / policy_reward_np.size

                            time_dif = self.data_helper.get_time_dif(start_time)
                            msg = 'Iter: {0:>6}, Train Loss: {1:>8.2}, Train Acc: {2:>7.2%},' \
                                  + 'Time: {3}'

                            print(msg.format(total_batch, loss_train, acc_train, time_dif))

                            if total_batch % (self.config_util.print_per_batch * 6) == 0:
                                loss_val, acc_val = self.evaluate(policy_session, global_session, my_agent, global_model,
                                                                  mention_test, entity_test, fea_test, entity_embedd_test, y_test)

                                # 保存最好结果
                                if acc_val > best_acc_val:
                                    best_acc_val = acc_val
                                    last_improved = total_batch
                                    policy_saver.save(sess=policy_session, save_path=self.config_util.save_policy_dir)
                                    improved_str = '*'
                                else:
                                    improved_str = ''

                                time_dif = self.data_helper.get_time_dif(start_time)
                                msg = 'Iter: {0:>6}, Train Loss: {1:>8.2}, Train Acc: {2:>7.2%},' \
                                      + ' Val Loss: {3:>8.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'

                                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif,
                                                 improved_str))

                        # 对loss进行优化
                        policy_session.run(my_agent.optim, feed_dict=seq_feed_dict)
                        total_batch += 1

                        # 验证集正确率长期不提升，提前结束训练
                        if total_batch - last_improved > self.config_util.local_require_improvement:
                            print("No optimization for a long time, auto-stopping...")
                            early_stop_flag = True
                            break

                    # early stopping
                    if early_stop_flag:
                        break

        global_session.close()
        policy_session.close()

    def test_agent(self):
        """
        测试训练好的模型
        :return:
        """
        print "Test Agent..."

        # 加载测试数据
        mention_test, entity_test, fea_test, entity_embedd_test, y_test = self.load_policy_test_data()

        # 创建全局模型的graph
        global_graph = tf.Graph()
        global_session = tf.Session(graph=global_graph, config=self.gpu_config)
        # 加载全局模型
        with global_session.as_default():
            with global_graph.as_default():
                global_model = GlobalModel(self.config_util)
                global_session.run(tf.global_variables_initializer())
                global_saver = tf.train.Saver()
                global_saver.restore(sess=global_session, save_path=self.config_util.save_global_dir)

        # 创建policy模型的graph
        policy_graph = tf.Graph()
        policy_session = tf.Session(graph=policy_graph, config=self.gpu_config)
        with policy_session.as_default():
            with policy_graph.as_default():
                my_agent = Agent(self.config_util)
                policy_session.run(tf.global_variables_initializer())
                policy_saver = tf.train.Saver()
                policy_saver.restore(sess=policy_session, save_path=self.config_util.save_policy_dir)
                loss_val, acc_val = self.evaluate(policy_session, global_session, my_agent, global_model,
                                                  mention_test, entity_test, fea_test, entity_embedd_test, y_test)

                print loss_val, acc_val


if __name__ == "__main__":
    data_helper = DataUtil()
    global_config_util = GlobalConfigUtil()
    agent_process = AgentProcess(global_config_util, data_helper)
    agent_process.train_agent()
