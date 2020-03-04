# encoding:utf-8

import json
import pre_data_config


class FileUtil(object):
    """
    文件工具类
    """

    def __init__(self):
        pass


    def combine_file(self, path_list, combine_path):
        """
        合并多个文件
        :param path_list: 文件名集合
        :param combine_path: 合并后文件
        :return:
        """
        with open(combine_path, "w") as combine_file:
            for file_path in path_list:
                with open(file_path, "r") as train_file:
                    for item in train_file:
                        item = item.strip().decode("utf-8")

                        combine_file.write(item.encode("utf-8") + "\n")

    def extract_testA(self, testAB_source_path, testA_source_path):
        """
        抽取出testA数据
        :param testAB_source_path:
        :param testA_source_path:
        :return:
        """
        with open(testAB_source_path, "r") as testa_testb_file:
            with open(testA_source_path, "w") as testa_file:
                doc_count = 0
                for item in testa_testb_file:
                    item = item.strip().decode("utf-8")

                    # 文章开头标志
                    if item.__contains__("DOCSTART"):
                        doc_count += 1
                        if doc_count <= 214:
                            testa_file.write(item.encode("utf-8") + "\n")
                        continue

                    if doc_count <= 214:
                        testa_file.write(item.encode("utf-8") + "\n")



    def extract_testB(self, testAB_source_path, testB_source_path):
        """
        抽取出testB数据
        :param testAB_source_path:
        :param testB_source_path:
        :return:
        """
        with open(testAB_source_path, "r") as testa_testb_file:
            with open(testB_source_path, "w") as testb_file:
                doc_count = 0
                for item in testa_testb_file:
                    item = item.strip().decode("utf-8")

                    # 文章开头标志
                    if item.__contains__("DOCSTART"):
                        doc_count += 1
                        if doc_count > 214:
                            testb_file.write(item.encode("utf-8") + "\n")
                        continue

                    if doc_count > 214:
                        testb_file.write(item.encode("utf-8") + "\n")

    def filter_low_pv(self, source_path, filter_pv_path, pv_value):
        """
        过滤文件中pv较低的mention
        :param source_path:
        :param filter_pv_path:
        :param pv_value: pv分界
        :return:
        """
        with open(source_path, "r") as source_file:
            with open(filter_pv_path, "w") as filter_pv_file:
                for item in source_file:
                    item = item.strip().decode("utf-8")
                    content_split = item.split("\t")

                    # 一篇文章的mention对应所有候选实体信息
                    if content_split[0].__contains__("__json_result__"):

                        json_str = content_split[0].split("__json_result__:")[-1]

                        try:
                            mention_candidate_json = json.loads(json_str)
                            for mention_name, candidate_entity_list in mention_candidate_json.items():
                                if mention_name == "mention_num":
                                    continue

                                for candidate_entity in candidate_entity_list:
                                    if candidate_entity.has_key("flag") and candidate_entity.has_key("pv"):
                                        # 正例为1，负例为0
                                        if candidate_entity["flag"] == 1:
                                            # 小于设定的pv值时删除当前mention
                                            if candidate_entity["pv"] < pv_value:
                                                mention_candidate_json.pop(mention_name)

                            filter_pv_file.write("__json_result__:".encode("utf-8") +
                                                 json.dumps(mention_candidate_json, ensure_ascii=False).encode("utf-8") + "\n")

                        except:
                            pass


                    else:
                        filter_pv_file.write(item.encode("utf-8") + "\n")


    def static_doc(self, data_path):
        """
        统计文档相关信息
        :param data_path:
        :return:
        """

        doc_count = 0

        with open(data_path, "r") as data_file:
            for item in data_file:
                item = item.strip().decode("utf-8")

                # 一篇文章的mention对应所有候选实体信息
                if item.__contains__("DOCSTART"):
                    doc_count += 1

            print doc_count





if __name__ == "__main__":
    file_util = FileUtil()

    # combine_path_list = [pre_data_config.ace_test_path, pre_data_config.msnbc_test_path,
    #                      pre_data_config.aquaint_test_path, pre_data_config.clueweb_test_path,
    #                      pre_data_config.wikipedia_test_path, pre_data_config.aida_train_path,
    #                      pre_data_config.aida_test_ab_path]

    # combine_path_list = [pre_data_config.aida_train_testa_cut_group_path, pre_data_config.wiki_crawl_cut_group]
    # combine_train_path = pre_data_config.combine_train_group_path
    # file_util.combine_file(combine_path_list, combine_train_path)

    file_util.static_doc("/Users/xie/el_data/wiki_crawl_origin_data")








