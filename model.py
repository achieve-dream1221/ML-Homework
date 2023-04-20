#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/19 18:22
# @Author  : achieve_dream
# @File    : model.py
# @Software: Pycharm
import cv2
import numpy as np
from abc import ABC, abstractmethod


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
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(xs, x):
        return np.exp(x) / np.sum(np.exp(xs))

    @abstractmethod
    def compute(self, *args, **kwargs):
        """
        计算方法结果
        :return: None
        """
        ...

    @abstractmethod
    def plot(self, *args, **kwargs):
        """
        绘制图形
        :return: None
        """
        ...
