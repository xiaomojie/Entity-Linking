# encoding: utf-8

import json
import xgboost as xgb
import numpy as np
from collections import defaultdict
import heapq

class XgboostRank(object):

    def __init__(self, model_path):
        self.model = None
        self.param = {'booster': 'gbtree', 'max_depth': 5, 'eta': 0.01, 'silent': 1, 'objective': 'rank:pairwise',
                      'gamma': 0.2, 'lambda': 700, 'subsample': 0.8, 'seed': 1}
        self.num_round = 1000

        self.model_path = model_path

    def format_data(self, source_data_path, format_data_path, group_file_path):
        """
        格式化为xgboost处理的数据
        :param source_data_path:
        :param format_data_path:
        :param group_file_path:
        :return:
        """

        qid = 0
        with open(source_data_path, "r") as source_data_file:
            with open(format_data_path, "w") as format_data_file:
                with open(group_file_path, "w") as group_data_file:
                    for item in source_data_file:
                        item = item.strip().decode("utf-8")

                        # 文章开头标志
                        if item.__contains__("DOCSTART"):
                            group_data_file.write(item.encode("utf-8") + "\n")
                            continue

                        mention_name, mention_sent, mention_candidate_entity = item.split("\t")
                        candidate_entity_list = json.loads(mention_candidate_entity)

                        # 所有mention
                        qid += 1
                        for entity in candidate_entity_list:
                            label = 2
                            # 正例
                            if entity["flag"] == 1:
                                label = 1
                            # 负例
                            elif entity["flag"] == 0:
                                label = 2
                            feature = entity["feature"]

                            fea_list = [fea_name + ":" + str(fea_val) for fea_name, fea_val in feature.items()]

                            format_data_file.write(str(label).encode("utf-8") + " qid:" + str(qid).encode("utf-8")
                                                   + " " + " ".join(fea_list).encode("utf-8") + "\t" + "#" + "\t" +
                                                   mention_sent.encode("utf-8") + "\t" +
                                                   json.dumps(entity, ensure_ascii=False).encode("utf-8") + "\n")

                        group_data_file.write(str(len(candidate_entity_list)).encode("utf-8") + "\n")

                        # 含有歧义mention
                        # if len(candidate_entity_list) > 1:
                        #     qid += 1
                        #     for entity in candidate_entity_list:
                        #         label = 2
                        #         if entity["flag"] == 1:
                        #             label = 1
                        #         elif entity["flag"] == 0:
                        #             label = 2
                        #         feature = entity["feature"]
                        #
                        #         fea_list = [fea_name + ":" + str(fea_val) for fea_name, fea_val in feature.items()]
                        #
                        #         format_data_file.write(str(label).encode("utf-8") + " qid:" + str(qid).encode("utf-8")
                        #                                + " " + " ".join(fea_list).encode("utf-8") + "\t" + "#" + "\t" +
                        #                                mention_sent.encode("utf-8") + "\t" +
                        #                                json.dumps(entity, ensure_ascii=False).encode("utf-8") + "\n")
                        #
                        #     group_data_file.write(str(len(candidate_entity_list)).encode("utf-8") + "\n")


    def read_data_from_file(self, file_name):
        """
        读取数据
        :param file_name:
        :return:
        """
        y_list = []
        x_list = []
        with open(file_name) as fp:
            for line in fp:
                uline = line.strip().decode('utf-8')
                features = uline.split('\t')[0]
                ulineList = features.split(' ')
                _y = np.int(ulineList[0])
                _x = [np.float(x.split(':')[1]) for x in ulineList[2:]]
                y_list.append(_y)
                x_list.append(_x)
        return np.array(y_list), np.array(x_list)

    def read_group(self, file_name):
        """
        读取rank数据对应的group文件
        :param file_name:
        :return:
        """
        group_list = []
        for line in open(file_name):
            uline = line.strip().decode('utf-8')
            # 文章开头标志
            if uline.__contains__("DOCSTART"):
                continue

            group_count = np.int(uline)
            group_list.append(group_count)
        return np.array(group_list)

    def train_models(self, feature_path, group_path):
        """
        训练模型
        :param feature_path:
        :param group_path:
        :return:
        """
        y, x = self.read_data_from_file(feature_path)
        group_list = self.read_group(group_path)
        dtrain = xgb.DMatrix(x, label=y)
        dtrain.set_group(group_list)
        self.model = xgb.train(self.param, dtrain, self.num_round)
        self.model.save_model(self.model_path)
        self.model.dump_model(self.model_path + '.dump.txt')
        return

    def load_rank_model(self, model_path=None):
        """
        加载保存的模型
        :param model_path:
        :return:
        """
        _model_path = model_path if model_path else self.model_path
        self.model = xgb.Booster()
        self.model.load_model(_model_path)
        return self.model

    def compute_precision(self, y, preds, group_list):
        num = len(group_list)
        correct = 0
        i = 0
        j = group_list[0]
        group_index = 0
        while 1:
            group_index += 1
            if group_index >= num:
                break
            _y_list = y[i:j]
            _preds_list = preds[i:j]
            _y_index = _y_list.index(min(_y_list))
            _preds_index = _preds_list.index(min(_preds_list))
            if _y_index == _preds_index:
                correct += 1
            # else:
            #     self.badcase_file.write("label: " + str(i + _y_index) + "\tpredict: " + str(i + _preds_index) + "\n")

            i = j
            j = i + group_list[group_index]
        preci = float(correct) / num
        return preci

    def predict_from_file(self, feature_path, group_path):
        """
        计算预测准确率
        :param feature_path:
        :param group_path:
        :return:
        """
        y, x = self.read_data_from_file(feature_path)
        dtest = xgb.DMatrix(x, label=y)
        group_list = self.read_group(group_path)
        dtest.set_group(group_list)
        preds = self.model.predict(dtest)

        # 计算group准确率
        precision = self.compute_precision(list(y), list(preds), list(group_list))
        return precision

    def cut_candidate(self, feature_path, group_path, cut_candidate_path, cut_group_path, max_num):
        """
        过滤多余的候选实体
        :param feature_path: rank格式数据文件
        :param group_path: rank对应group文件
        :param max_num: 最大候选实体数量
        :param cut_candidate_path: 过滤多余候选实体后的文件路径
        :param cut_group_path: 过滤多余候选实体后的group路径
        :return:
        """
        y, x = self.read_data_from_file(feature_path)
        dtest = xgb.DMatrix(x, label=y)
        group_list = self.read_group(group_path)
        dtest.set_group(group_list)
        preds = self.model.predict(dtest)

        print("all mention: " + str(len(group_list)))

        # 保存的候选实体index列表
        save_index_list = []
        # 正确的候选实体index列表
        positive_index_list = []
        # 保存的group_size列表
        save_group_list =[]
        num = len(group_list)
        i = 0
        j = group_list[0]
        group_index = 0
        group_size = group_list[group_index]

        while 1:
            _y_list = list(y[i:j])
            _preds_list = list(preds[i:j])

            # 正确候选实体下标
            _y_index = _y_list.index(min(_y_list))
            positive_index_list.append(i+_y_index)

            # 根据预测结果对候选实体排序，预测正确的排在首位
            tmp_dict = {}
            for index, val in enumerate(_preds_list):
                tmp_dict[index] = val
            small_index_list = [item[0] for item in sorted(tmp_dict.items(), key=lambda x: x[1])]

            # 确定group中最小的n个候选实体
            if group_size > max_num:
                small_index_list = small_index_list[:max_num]

                # 正确候选实体未召回时将候选实体召回
                if _y_index not in set(small_index_list):
                    small_index_list.pop(-1)
                    small_index_list.append(_y_index)

                save_index_list.extend([i + index for index in small_index_list])
                save_group_list.append(max_num)
            # 候选实体数量小于max_num且大于1
            elif group_size > 1:
                save_index_list.extend([i + index for index in small_index_list])
                # 添加负例index
                add_neg_index = group_size - 1
                for tmp_index in range(group_size-1, -1, -1):
                    if tmp_index != _y_index:
                        add_neg_index = tmp_index
                        break

                save_index_list.extend([i + add_neg_index for time in range(max_num - group_size)])
                save_group_list.append(max_num)
            # 只有1个候选实体暂时删掉
            else:
                pass

            group_index += 1
            if group_index >= num:
                break

            i = j
            group_size = group_list[group_index]
            j = i + group_size

        print(len(save_index_list), sum(save_group_list))
        print("disambiguation mention num: " + str(len(save_group_list)))

        # 计算候选实体召回率
        # count = 0
        # for positive_index in positive_index_list:
        #     if positive_index in save_index_set:
        #         count += 1
        # print "recall is: {}".format(float(count) / len(positive_index_list))

        # 过滤后的实体文件
        entity_all_list = []
        with open(feature_path, "r") as source_feature_file:
            with open(cut_candidate_path, "w") as cut_candidate_file:
                for item in source_feature_file:
                    item = item.strip().decode("utf-8")
                    entity_all_list.append(item)

                for save_index in save_index_list:
                    cut_candidate_file.write(entity_all_list[save_index].encode("utf-8") + "\n")


        # 过滤后的group文件
        group_sum = 0
        with open(group_path, "r") as group_file:
            with open(cut_group_path, "w") as cut_group_file:
                for item in group_file:
                    item = item.strip().decode("utf-8")
                    # 文章开头标志
                    if item.__contains__("DOCSTART"):
                        cut_group_file.write(item.encode("utf-8") + "\n")
                        continue

                    if int(item) > 1:
                        group_sum += max_num
                        cut_group_file.write(str(max_num).encode("utf-8") + "\n")

        print("group_sum:{0}, save_group_list:{1}".format(group_sum, sum(save_group_list)))

    def cul_recall(self, feature_path, group_path, max_num):
        """
        计算候选召回率
        :return:
        """
        y, x = self.read_data_from_file(feature_path)
        dtest = xgb.DMatrix(x, label=y)
        group_list = self.read_group(group_path)
        dtest.set_group(group_list)
        preds = self.model.predict(dtest)

        # 保存的候选实体index列表
        save_index_list = []
        # 正确的候选实体index列表
        positive_index_list = []
        # 保存的group_size列表
        save_group_list = []
        num = len(group_list)
        i = 0
        j = group_list[0]
        group_index = 0
        group_size = group_list[group_index]

        while 1:
            _y_list = list(y[i:j])
            _preds_list = list(preds[i:j])

            # 正确候选实体下标
            _y_index = _y_list.index(min(_y_list))
            positive_index_list.append(i + _y_index)

            # 根据预测结果对候选实体排序，预测正确的排在首位
            tmp_dict = {}
            for index, val in enumerate(_preds_list):
                tmp_dict[index] = val
            small_index_list = [item[0] for item in sorted(tmp_dict.items(), key=lambda x: x[1])]

            # 确定group中最小的n个候选实体
            if group_size > max_num:
                small_index_list = small_index_list[:max_num]

                save_index_list.extend([i + index for index in small_index_list])
                save_group_list.append(max_num)
            # 候选实体数量小于max_num且大于1
            elif group_size > 1:
                save_index_list.extend([i + index for index in small_index_list])
                save_group_list.append(group_size)
            else:
                save_index_list.extend([i + _y_index])
                save_group_list.append(1)

            group_index += 1
            if group_index >= num:
                break

            i = j
            group_size = group_list[group_index]
            j = i + group_size

        print(len(save_index_list), sum(save_group_list))

        # 计算候选实体召回率
        save_index_set = set(save_index_list)
        count = 0
        for positive_index in positive_index_list:
            if positive_index in save_index_set:
                count += 1
        print("recall is: {}".format(float(count) / len(positive_index_list)))

    def cul_fea_weight(self):
        """
        计算每个feature的权重信息
        :return:
        """
        model = self.load_rank_model(self.model_path)
        importance = model.get_fscore()

        val_list = []
        for fea, val in importance.items():
            val_list.append(val)

        fea_weight = [round(float(item)/sum(val_list), 3) for item in val_list]
        print(fea_weight)


    def extract_testB(self, testa_testb_candidate_path, testa_testb_group_path, testb_candidate_path, testb_group_path):
        """
        抽取testB数据
        :param testa_testb_candidate_path:
        :param testa_testb_group_path:
        :param testb_candidate_path:
        :param testb_group_path:
        :return:
        """
        with open(testa_testb_group_path, "r") as testa_testb_group_file:
            with open(testb_group_path, "w") as testb_group_file:
                doc_count = 0
                entity_num = 0
                for item in testa_testb_group_file:
                    item = item.strip().decode("utf-8")

                    # 文章开头标志
                    if item.__contains__("DOCSTART"):
                        doc_count += 1
                        if doc_count > 214:
                            testb_group_file.write(item.encode("utf-8") + "\n")
                        continue

                    if doc_count <= 214:
                        entity_num += int(item)
                    else:
                        testb_group_file.write(item.encode("utf-8") + "\n")

        print(entity_num)
        with open(testa_testb_candidate_path, "r") as testa_testb_candidate_file:
            with open(testb_candidate_path, "w") as testb_candidate_file:
                count = 0
                for item in testa_testb_candidate_file:
                    count += 1
                    if count > entity_num:
                        item = item.strip().decode("utf-8")

                        testb_candidate_file.write(item.encode("utf-8") + "\n")

    def test_mention_num(self, test_group_path):
        """
        测试mention数量和group中数量是否一致
        :param test_group_path:
        :return:
        """

        all_doc_list = []
        doc_mention_list = []
        with open(test_group_path, "r") as group_file:
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

        print(len(all_doc_list))
        print("group_mention_num:{0}".format(sum([sum(item) for item in all_doc_list])))

    def pv_static(self, format_path):
        """
        对正确目标实体的pv值进行统计
        :param format_path:
        :return:
        """

        y, x = self.read_data_from_file(format_path)
        print(len(x))

        pv_list = []
        for fea_list, y_item in zip(x, y):
            if y_item == 1:
                pv = fea_list[1]
                pv_list.append(pv)

        pv_list = sorted(pv_list)

        print(pv_list[1000])
        print(pv_list[2000])
        print(pv_list[3000])
        print(pv_list[3400])
        print(pv_list[-1])

    def pv_divide(self, format_path, group_path, prefix_pv_path):
        """
        按照正确目标实体的pv值对文件进行划分
        :return:
        """

        pv_file_list = [open(prefix_pv_path + "pv" + str(index), "w") for index in range(6)]
        pv_group_file_list = [open(prefix_pv_path + "group" + str(index), "w") for index in range(6)]
        pv_val_list = [0.01, 0.1, 1, 5, 10, 100]

        group_list = self.read_group(group_path)
        all_data_list = []

        with open(format_path, "r") as format_file:
            for item in format_file:
                item = item.strip().decode("utf-8")
                all_data_list.append(item)

        begin = 0
        group_index = 0
        end = group_list[group_index]
        num = len(group_list)

        while 1:
            group_data_list = all_data_list[begin:end]
            for data in group_data_list:
                line = data.split('\t')[0]
                line_list = line.split(' ')
                label = np.int(line_list[0])
                fea_list = [np.float(x.split(':')[1]) for x in line_list[2:]]

                # 正例
                if label == 1:
                    pv = fea_list[1]

                    for index in range(6):
                        if pv <= pv_val_list[index] and pv != 0:
                            for line in group_data_list:
                                pv_file_list[index].write(line.encode("utf-8") + "\n")
                            pv_group_file_list[index].write(str(len(group_data_list)) + "\n")
                            break
                    break

            group_index += 1
            if group_index >= num:
                break

            begin = end
            end = begin + group_list[group_index]

    def controller_train(self):
        source_data_path = "/Users/xie/el_data/aida_train_testa/aida_train_testa_feature"

        format_data_path = "/Users/xie/el_data/aida_train_testa/aida_train_testa_format"

        group_file_path = "/Users/xie/el_data/aida_train_testa/aida_train_testa_group"

        cut_candidate_path = "/Users/xie/el_data/aida_train_testa/aida_train_testa_cut_candidate"

        cut_group_path = "/Users/xie/el_data/aida_train_testa/aida_train_testa_cut_group"

        self.format_data(source_data_path, format_data_path, group_file_path)
        print("format xgboost data success")

        self.train_models(format_data_path, group_file_path)
        print(self.predict_from_file(format_data_path, group_file_path))

        self.load_rank_model()
        self.cut_candidate(format_data_path, group_file_path, cut_candidate_path, cut_group_path, max_num=5)

    def controller_test(self):
        source_data_path = "/Users/xie/el_data/aida_testb/aida_testb_source_feature"

        format_data_path = "/Users/xie/el_data/aida_testb/aida_testb_rank_format"

        group_file_path = "/Users/xie/el_data/aida_testb/aida_testb_group"

        cut_candidate_path = "/Users/xie/el_data/aida_testb/aida_testb_cut_candidate"

        cut_group_path = "/Users/xie/el_data/aida_testb/aida_testb_cut_group"

        # self.format_data(source_data_path, format_data_path, group_file_path)
        # print "format xgboost data success"

        # self.load_rank_model()
        # print self.predict_from_file(format_data_path, group_file_path)

        # self.load_rank_model()
        # self.cut_candidate(format_data_path, group_file_path, cut_candidate_path, cut_group_path, max_num=5)

        prefix_path = "/Users/xie/el_data/aida_testb/pv_list/"
        # self.pv_divide(format_data_path, group_file_path, prefix_path)
        self.load_rank_model()
        for index in range(1):
            self.cut_candidate(prefix_path + "pv" + str(index), prefix_path + "group" + str(index),
                               prefix_path + "cut_candidate" + str(index), prefix_path + "cut_group" + str(index),
                               max_num=5)





if __name__ == "__main__":
    model_path = "/Users/xie/el_data/aida_train/xgboost_model"
    xgboost_rank = XgboostRank(model_path)
    xgboost_rank.controller_test()
