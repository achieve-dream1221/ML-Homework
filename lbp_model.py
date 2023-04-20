#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/18 23:18
# @Author  : achieve_dream
# @File    : lbp_model.py
# @Software: Pycharm
import numpy as np
from skimage.feature import local_binary_pattern
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

    def compute(self) -> tuple[np.ndarray, np.ndarray]:
        """
        计算所有数据集的lbp值, 然后算出后面9列对第一列的距离, 并用sigmoid进行归一化
        :return: None
        """
        # 存储第一列的lbp, 方便后面的数据集和它进行比较
        lbp1_list = [local_binary_pattern(self.read_img(col_1), self.__n_points, self.__radius, method='ror') for col_1
                     in self.dataset[:, 0]]
        predicts = []
        target_labels = []
        # 计算后9列到第1列的距离, 30 * 9 * 30
        row_index = 0
        # 图像有多少分类
        class_nums = self.img_nums // 10
        for cols in self.dataset[:, 1:]:  # class_nums 行
            # 9 * 30, 真实标签[ 1, 0, 0, 0 ..., 0] * 9 (9列属于同一个类)
            target_labels_rows = [*[0] * row_index, 1, *[0] * (class_nums - 1 - row_index)] * 9
            # 展开为一行
            target_labels.extend(target_labels_rows)
            row_index += 1
            for col in cols:  # 9 列
                lbp2 = local_binary_pattern(self.read_img(col), self.__n_points, self.__radius, method='ror')
                # 先计算lbp的欧式距离, 然后通过softmax预测出这10个类别的概率
                predict_scores = self.softmax(np.array([np.linalg.norm(lbp - lbp2) for lbp in lbp1_list]))
                # 9 * 30, 预测标签
                predicts.extend(predict_scores)
        return np.array(target_labels), np.array(predicts)

    def run(self):
        targets, predicts = self.compute()
        np.save("target_labels", targets)
        np.save("lbp_predicts", predicts)
        self.plot(targets, predicts)


if __name__ == '__main__':
    model = LBPModel(img_nums=300)
    model.plot(np.load("target_labels.npy"), np.load("lbp_predicts.npy"))
