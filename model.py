#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/19 18:22
# @Author  : achieve_dream
# @File    : model.py
# @Software: Pycharm
from random import sample

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

    @staticmethod
    def pure_load_dataset(img_nums: int):
        """
        只进行数字处理, 不进行IO操作
        :return: dataset(选取的图像的下标)
        """
        # 对图像进行随机采样
        result = []
        for i in range(1, 301, 10):
            samples = [samples for samples in
                       sample(range(i, i + 10), k=5)]
            result.extend(samples)
        # return [i for i in range(1, img_nums + 1)]
        return result

    @staticmethod
    def read_img(img_index: int) -> np.ndarray:
        return cv2.imread("dataset/" + str(img_index).rjust(5, '0') + ".bmp", cv2.IMREAD_GRAYSCALE)

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
