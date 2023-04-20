#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/19 18:47
# @Author  : achieve_dream
# @File    : pca_model.py
# @Software: Pycharm
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc

from model import Model
from sklearn.decomposition import PCA


class PCAModel(Model):
    # 测试集,训练集的lbp和图像位置下标
    train_pca_list, test_pca_list, train_num_list, test_num_list = [], [], [], []
    # 图像的lbp值
    pca_list = None
    pca_score_list = None
    real_labels = []

    def __init__(self):
        super().__init__()

    def compute(self):
        self.pca_list = [PCA().fit_transform(self.read_img(img_index)) for img_index in self.dataset]
        # for img_index in self.dataset:
        #     img = self.read_img(img_index)
        #     # 计算PCA的主成分和均值
        #     mean, eigenvectors = cv2.PCACompute(img, mean=None)
        #     result = cv2.PCAProject(img, mean, eigenvectors)

    def split_dataset(self):
        """
        按照3: 2, 分割数据集
        :return: None
        """
        for i in range(0, 100, 5):
            self.train_num_list.extend(self.dataset[i:i + 3])
            self.test_num_list.extend(self.dataset[i + 3:i + 5])
            self.train_pca_list.extend(self.pca_list[i:i + 3])
            self.test_pca_list.extend(self.pca_list[i + 3: i + 5])

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
        for train_lbp in self.train_pca_list:
            for test_lbp in self.test_pca_list:
                distance_lbp_list.append(np.linalg.norm(train_lbp - test_lbp))
        # 归一化处理
        d_min = np.min(distance_lbp_list)
        d_max = np.max(distance_lbp_list)
        d_d = d_max - d_min
        self.pca_score_list = [1 - (distance - d_min) / d_d for distance in distance_lbp_list]

    def plot(self):
        fpr, tpr, _ = roc_curve(self.real_labels, self.pca_score_list)
        roc_auc_pca = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='ROC_pca curve (area = %0.2f)' % roc_auc_pca)
        plt.plot([0, 1], [0, 1])
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig('pca-roc.svg')
        plt.show()

    def run(self):
        self.compute()
        self.split_dataset()
        self.get_real_label()
        self.compute_distance()
        self.plot()


if __name__ == '__main__':
    model = PCAModel()
    model.run()
