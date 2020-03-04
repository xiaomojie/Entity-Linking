# encoding:utf-8

import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


class FigureUtil(object):
    """
    画图工具类
    """

    def histogram(self):
        """
        单一数据的柱状图
        :return:
        """
        x_label = ["< 0.01", "0.01 ~ 0.1", "0.1 ~ 1", "1 ~ 5", "5 ~ 10", "> 10"]
        y_list = [0.92, 0.88, 0.87, 0.96, 0.97, 0.99]
        x_list = range(len(y_list))

        bar_width = 0.25

        plt.bar(left=x_list, height=y_list, alpha=0.8, color='dimgray', width=0.5, edgecolor="white")
        ylim(0.7, 1.0)
        plt.margins(0.05)
        plt.xticks([item+bar_width for item in x_list], x_label)
        plt.xlabel("Pageview/million")
        plt.ylabel("Micro F1")
        plt.show()

    def double_histogram(self):
        """
        具有对比数据的直方图
        :return:
        """

        # names = ("", ""no dynamic ranker", "static and dynamic ranker")
        # subjects = ("AIDA-TestA", "ACE2004", "RSS500", "Reuters128")
        # scores = ((0.92, 0.878, 0.762, 0.81), (0.934, 0.916, 0.79, 0.847))

        # names = ("SF", "SE", "EH")
        # subjects = ("MSNBC", "ACE2004", "RSS500", "Reuters128")
        # scores = ((0.917, 0.927, 0.900, 0.904), (0.552, 0.482, 0.459, 0.386), (0.010, 0.040, 0.115, 0.249))

        names = ("GCN", "GAT", "SeqGAT")
        subjects = ("AIDA-A", "AIDA-B", "ACE2004", "Reuters128")
        scores = ((0.785, 0.795, 0.862, 0.698), (0.80, 0.81, 0.875, 0.70), (0.83, 0.84, 0.89, 0.71))

        # names = ("no topic filter", "topic filter")
        # subjects = ("MSNBC", "ACE2004", "WNED-CWEB", "WNED-CWEB")
        # scores = ((0.91, 0.864, 0.756, 0.80), (0.934, 0.916, 0.79, 0.847))

        # names = ("no global encoder", "global encoder")
        # subjects = ("MSNBC", "ACE2004", "WNED-CWEB", "WNED-CWEB")
        # scores = ((0.912, 0.874, 0.758, 0.80), (0.934, 0.916, 0.79, 0.847))

        index = np.arange(len(scores[0]))
        bar_width = 0.25


        #第一个柱状图
        # rects1 = plt.bar(left=index, height=scores[0], width=bar_width, color='deepskyblue', label=names[0])
        # rects2 = plt.bar(left=index + bar_width, height=scores[1], width=bar_width, color='blueviolet', label=names[1])

        # 第二个柱状图
        # rects1 = plt.bar(left=index, height=scores[0], width=bar_width, color='deepskyblue', label=names[0])
        # rects2 = plt.bar(left=index + bar_width, height=scores[1], width=bar_width, color='coral', label=names[1])

        #第三个柱状图
        rects1 = plt.bar(left=index, height=scores[0], width=bar_width, color='b', label=names[0])
        rects2 = plt.bar(left=index + bar_width, height=scores[1], width=bar_width, color='g', label=names[1])
        rects3 = plt.bar(left=index + 2*bar_width, height=scores[2], width=bar_width, color='r', label=names[2])


        # X轴标题
        plt.xticks(index + 1.5*bar_width, subjects)
        # Y轴范围
        ylim(0.5, 1.0)
        plt.margins(0.05)
        # 显示图例
        plt.legend()

        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 20,
                 }
        plt.ylabel("gold recall", font2)

        # 添加数据标签
        def add_labels(rects):
            for rect in rects:
                # height = rect.get_height()
                # plt.text(rect.get_x() + rect.get_width() / 2, height, height, ha='center', va='bottom')
                # 柱形图边缘用白色填充，纯粹为了美观
                rect.set_edgecolor('white')

        add_labels(rects1)
        add_labels(rects2)
        add_labels(rects3)

        plt.savefig("/Users/xie/Desktop/a.pdf")
        plt.show()

    def line_chart(self):
        """
        绘制折线图
        :return:
        """
        # 多折线图
        names = [str(i) for i in range(1, 21)]

        x = range(len(names))

        y_aida = [0.82, 0.858, 0.894, 0.916, 0.932, 0.94, 0.943, 0.945, 0.945, 0.947,
             0.944, 0.942, 0.94, 0.938, 0.935, 0.93, 0.925, 0.92, 0.918, 0.915]

        y_msnbc = [0.84, 0.874, 0.893, 0.917, 0.925, 0.934, 0.94, 0.943, 0.947, 0.945,
                  0.942, 0.936, 0.934, 0.932, 0.93, 0.928, 0.926, 0.926, 0.926, 0.92]

        y_aquaint = [0.81, 0.835, 0.852, 0.861, 0.87, 0.878, 0.883, 0.888, 0.892, 0.895,
                  0.892, 0.892, 0.89, 0.886, 0.882, 0.88, 0.879, 0.876, 0.873, 0.873]

        y_cweb = [0.70, 0.73, 0.75, 0.77, 0.783, 0.788, 0.793, 0.796, 0.798, 0.80,
                  0.80, 0.794, 0.79,  0.786, 0.783, 0.783, 0.78, 0.78, 0.778, 0.777]

        y_cwiki = [0.725, 0.76, 0.788, 0.81, 0.821, 0.834, 0.845, 0.852, 0.856, 0.86,
                  0.858, 0.855, 0.851, 0.848, 0.844, 0.841, 0.839, 0.837, 0.835, 0.832]

        # 让图例生效
        plt.plot(x, y_aida, marker='o', color="r", lw=2, label="AIDA-B")
        # plt.plot(x, y_msnbc, marker='o', color='red', label='MSNBC')
        plt.plot(x, y_aquaint, marker='s', color='k', label='AQUAINT')
        plt.plot(x, y_cweb, marker='D', color='slategrey', label='WNED-CWEB')
        plt.plot(x, y_cwiki, marker='h', color='navy', label='WNED-CWIKI')


        plt.legend(loc='lower right')
        # # Y轴范围
        ylim(0.6, 1)
        plt.xticks(x, names)
        plt.margins(0)

        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 20,
                 }
        plt.xlabel("Number of k", font2)
        plt.ylabel("in-kb accuracy", font2)

        plt.grid()

        plt.show()

        ###################################################################################################

        # names = [str(i) for i in range(1, 16)]
        #
        # x = range(len(names))
        #
        # y_ac = [0.821, 0.89, 0.926, 0.947, 0.944, 0.936, 0.932, 0.93, 0.933, 0.926,
        #         0.924, 0.92, 0.917, 0.917, 0.92]
        #
        # y_pg = [0.663, 0.742, 0.819, 0.857, 0.895, 0.917, 0.924, 0.937, 0.943, 0.94,
        #         0.935, 0.93, 0.926, 0.922, 0.919]
        #
        # # 让图例生效
        # plt.plot(x, y_ac, marker='o', color="r", lw=2, label="Actor-Critic Network")
        # plt.plot(x, y_pg, marker='s', color='k', label='Policy Gradient')
        #
        #
        # plt.legend(loc='lower right')
        # # # Y轴范围
        # ylim(0.6, 1)
        # plt.xticks(x, names)
        # plt.margins(0)
        #
        # font2 = {'family': 'Times New Roman',
        #          'weight': 'normal',
        #          'size': 20,
        #          }
        # plt.xlabel("Epochs", font2)
        # plt.ylabel("in-kb accuracy", font2)
        #
        # plt.grid()
        #
        # plt.show()

        # names = [str(i/20.0) for i in range(0, 21)]
        #
        # x = range(len(names))
        # y = [0.93, 0.932, 0.935, 0.937, 0.932, 0.94, 0.935, 0.938, 0.94, 0.945, 0.943, 0.941, 0.947, 0.94, 0.942, 0.938, 0.934, 0.934, 0.936, 0.937, 0.94]
        # plt.plot(x, y, marker='o', mec='k', mfc='w', c="k", lw=2, label="")
        # # 让图例生效
        #
        # plt.legend()
        # # # Y轴范围
        # ylim(0.85, 1)
        # plt.xticks(x, names)
        # plt.margins(0)
        #
        # font2 = {'family': 'Times New Roman',
        #          'weight': 'normal',
        #          'size': 20,
        #          }
        # plt.xlabel("Value of $\\alpha$", font2)
        # plt.ylabel("in-kb accuracy", font2)
        #
        # plt.grid()
        # plt.show()

        # names = [str(i/10.0) for i in range(0, 21)]
        #
        # x = range(len(names))
        # y = [0.94, 0.942, 0.942, 0.94, 0.944, 0.94, 0.945, 0.938, 0.947, 0.945, 0.943, 0.941, 0.942, 0.935, 0.932, 0.93, 0.93, 0.925, 0.92, 0.915, 0.912]
        # plt.plot(x, y, marker='o', mec='k', mfc='w', c="k", lw=2, label="")
        # # 让图例生效
        #
        # plt.legend()
        # # # Y轴范围
        # ylim(0.85, 1)
        # plt.xticks(x, names)
        # plt.margins(0)
        #
        # font2 = {'family': 'Times New Roman',
        #          'weight': 'normal',
        #          'size': 20,
        #          }
        # plt.xlabel("Value of $\delta$", font2)
        # plt.ylabel("in-kb accuracy", font2)
        #
        # plt.grid()
        # plt.show()





if __name__ == "__main__":
    figure_util = FigureUtil()

    figure_util.double_histogram()