# encoding:utf-8


class CaseStudy(object):

    def filter_data(self, data_path):
        """
        过滤数据
        :param data_path: 数据路径
        :return:
        """
        filter_path = data_path + "_filter"
        with open(data_path, "r") as data_file:
            with open(filter_path, "w") as filter_file:
                for item in data_file:
                    item = item.strip().decode("utf-8")

                    line_list = item.split("\t")[:-1]
                    filter_file.write("\t".join(line_list).encode("utf-8") + "\n")


if __name__ == "__main__":
    data_path = "/Users/xie/el_data/case_study/testb_rank_mention"

    case_study = CaseStudy()
    case_study.filter_data(data_path)



