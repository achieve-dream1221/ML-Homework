#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/18 23:18
# @Author  : achieve_dream
# @File    : lbp_model.py
# @Software: Pycharm
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern
from sklearn.metrics import roc_curve, auc
from model import Model


class LBPModel(Model):
    """
    机器学习, 图像分类模型
    """

    def __init__(self, n_points: int = 8, radius: int = 3, img_nums: int = 300):
        """
        初始化对象
        :param n_points: lbp采样点数
        :param radius: lbp的采样半径
        :param img_nums: 加载数据集图片数量
        """
        super().__init__(img_nums)
        self.__n_points = n_points
        self.__radius = radius
        # 测试集,训练集的lbp和图像位置下标
        self.train_lbp_list, self.test_lbp_list, self.train_num_list, self.test_num_list = [], [], [], []
        # 图像的lbp值
        self.lbp_list = None
        self.lbp_score_list = None
        self.real_labels = []

    def compute(self):
        """
        计算所有数据集的lbp值
        :return: None
        """
        self.lbp_list = [local_binary_pattern(self.read_img(img_index), self.__n_points, self.__radius,
                                              method='ror') for img_index in self.dataset]

    def split_dataset(self):
        """
        按照3: 2, 分割数据集
        :return: None
        """
        for i in range(0, 100, 5):
            self.train_num_list.extend(self.dataset[i:i + 3])
            self.test_num_list.extend(self.dataset[i + 3:i + 5])
            self.train_lbp_list.extend(self.lbp_list[i:i + 3])
            self.test_lbp_list.extend(self.lbp_list[i + 3: i + 5])

    def get_real_label(self):
        for train_index in self.train_num_list:
            self.real_labels.extend(
                [1 if train_index // 10 == test_index // 10 else 0 for test_index in self.test_num_list])

    def compute_distance(self):
        """
        通过F范数计算距离即欧式距离, 并归一化
        :return: None
        """
        distance_lbp_list = []
        for train_lbp in self.train_lbp_list:
            for test_lbp in self.test_lbp_list:
                distance_lbp_list.append(np.linalg.norm(train_lbp - test_lbp))
        # 归一化处理
        d_min = np.min(distance_lbp_list)
        d_max = np.max(distance_lbp_list)
        d_d = d_max - d_min
        self.lbp_score_list = [1 - (distance - d_min) / d_d for distance in distance_lbp_list]

    def plot(self):
        fpr_lbp, tpr_lbp, thread_lbp = roc_curve(self.real_labels, self.lbp_score_list)
        roc_auc_lbp = auc(fpr_lbp, tpr_lbp)
        plt.plot(fpr_lbp, tpr_lbp, label='ROC_lbp curve (area = %0.2f)' % roc_auc_lbp)
        plt.plot([0, 1], [0, 1])
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig('roc.svg')
        plt.show()

    def run(self):
        self.compute()
        self.split_dataset()
        self.get_real_label()
        self.compute_distance()
        self.plot()


if __name__ == '__main__':
    model = LBPModel()
    model.run()
