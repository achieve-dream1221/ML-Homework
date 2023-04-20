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
        self.__distance_list = []  # 30 * 9
        # 图像的lbp值
        self.lbp_list = None
        self.lbp_score_list = None
        self.real_labels = []

    def compute(self):
        """
        计算所有数据集的lbp值, 然后算出后面9列对第一列的距离, 并用sigmoid进行归一化
        :return: None
        """
        # 存储第一列的lbp, 方便后面的数据集和它进行比较
        lbp1_list = [local_binary_pattern(self.read_img(col_1), self.__n_points, self.__radius, method='ror') for col_1
                     in self.dataset[:, 0]]
        # 计算后9列到第1列的距离, 30 * 9 * 30
        for cols in self.dataset[:, 1:]:
            for col in cols:
                lbp2 = local_binary_pattern(self.read_img(col), self.__n_points, self.__radius, method='ror')
                # 先计算lbp的欧式距离, 然后经过sigmoid输出为[0,1]之间的数
                rows = [self.sigmoid(np.linalg.norm(lbp - lbp2)) for lbp in lbp1_list]
                self.__distance_list.append(rows)

    def predict(self):
        distances = np.array(self.__distance_list)
        ...

    def split_dataset(self):
        """
        一个类别10张图片, 用1, 2列作为训练样本, 后8列作为测试样本, 分割数据
        :return: None
        """
        for i in range(0, self.img_nums + 1, 10):
            ...

    def get_real_label(self):
        ...

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
