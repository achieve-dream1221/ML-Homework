#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 17:43
# @Author  : achieve_dream
# @File    : wave_model.py
# @Software: Pycharm
import numpy as np
from pywt import waverec2, wavedec2

from model import Model


class HaarWaveModel(Model):

    def compute_haar(self, img_index: int):
        coeffs = wavedec2(self.read_img(img_index), 'haar')
        # return idwt2(dwt2(self.read_img(img_index), 'haar'), wavelet='haar')
        return waverec2(coeffs, 'haar')

    def compute(self, *args, **kwargs):
        """
        计算所有数据集的lbp值, 然后算出后面9列对第一列的距离, 并用sigmoid进行归一化
        :return: None
        """
        # 存储第一列的harr, 方便后面的数据集和它进行比较
        haar1_list = [self.compute_haar(col_1) for col_1 in self.dataset[:, 0]]
        predicts = []
        target_labels = []
        # 预测正确的个数
        positive_nums = 0
        # 计算后9列到第1列的距离, 30 * 9 * 30
        row_index = 0
        # 图像有多少分类
        class_nums = self.img_nums // 10
        for cols in self.dataset[:, 1:]:  # class_nums 行
            # 9 * 30, 真实标签[ 1, 0, 0, 0 ..., 0] * 9 (9列属于同一个类)
            target_labels_rows = [*[0] * row_index, 1, *[0] * (class_nums - 1 - row_index)] * 9
            # 展开为一行
            target_labels.extend(target_labels_rows)
            for col in cols:  # 9 列
                haar2 = self.compute_haar(col)
                # 先计算lbp的欧式距离, 然后通过softmax预测出这10个类别的概率
                predict_scores = self.softmax(np.array([np.linalg.norm(harr1 - haar2) for harr1 in haar1_list]))
                # 概率最大值的下标即分类标签
                if predict_scores.argmax() == row_index:
                    positive_nums += 1
                # 9 * 30, 预测标签
                predicts.extend(predict_scores)
            row_index += 1
        return np.array(target_labels), np.array(predicts), positive_nums / (class_nums * 9)

    def run(self):
        targets, predicts, accuracy = self.compute()
        print(f"Haar准确率: {round(accuracy, 2)}")
        np.save("target_labels", targets)
        np.save("wave_predicts", predicts)
        self.plot(targets, predicts, "wave_roc")


if __name__ == '__main__':
    model = HaarWaveModel(img_nums=300)
    model.run()
