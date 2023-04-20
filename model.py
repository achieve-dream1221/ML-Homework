#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/19 18:22
# @Author  : achieve_dream
# @File    : model.py
# @Software: Pycharm
import cv2
import numpy as np
from abc import ABC, abstractmethod

from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题


class Model(ABC):
    """
    机器学习, 图像分类模型
    """

    def __init__(self, img_nums: int):
        """
        构造方法, 初始化
        :param img_nums: 加载图片的数量
        """
        self.dataset = self.pure_load_dataset(img_nums)
        self.img_nums = img_nums

    @staticmethod
    def pure_load_dataset(img_nums: int) -> np.ndarray:
        """
        只进行数字处理, 不进行IO操作
        :return: dataset(选取的图像的下标)
        """
        # return np.array([[i for i in range(j, j + 10)] for j in range(1, img_nums + 1, 10)])
        return np.array([i for i in range(1, img_nums + 1)]).reshape(-1, 10)

    @staticmethod
    def read_img(img_index: int) -> np.ndarray:
        return cv2.imread("dataset/" + str(img_index).rjust(5, '0') + ".bmp", cv2.IMREAD_GRAYSCALE)

    @staticmethod
    def softmax(distance: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        根据图像的距离, 算它每个类别的概率
        :param distance: 一个图像到每个类别的距离
        :param eps: 一个很小的数, 防止距离为0而导致数据过大
        :return: 概率数组
        """
        # 由于距离和概率成反比, 因此取距离的倒数
        d = 1 / (distance + eps)
        e_x = np.exp(d - np.max(d))  # 防止指数爆炸
        return e_x / e_x.sum(axis=0)

    @abstractmethod
    def compute(self, *args, **kwargs):
        """
        计算方法结果
        :return: None
        """
        ...

    def plot(self, target_labels: np.ndarray, predict_scores: np.ndarray):
        """
        绘制图形
        :return: None
        """
        ...
        fpr, tpr, _ = roc_curve(target_labels, predict_scores)
        # roc面积
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='ROC 曲线 (面积 = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1])
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC曲线')
        plt.legend()
        plt.savefig('roc.svg')
        plt.show()
