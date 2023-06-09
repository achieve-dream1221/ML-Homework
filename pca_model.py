#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/19 18:47
# @Author  : achieve_dream
# @File    : pca_model.py
# @Software: Pycharm
import numpy as np

from model import Model
from sklearn.decomposition import PCA


class PCAModel(Model):
    def compute(self):
        pca = PCA()
        pca1_list = [pca.fit_transform(self.read_img(col_1)) for col_1 in self.dataset[:, 0]]
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
                pca2 = pca.fit_transform(self.read_img(col))
                # 先计算lbp的欧式距离, 然后通过softmax预测出这10个类别的概率
                predict_scores = self.softmax(np.array([np.linalg.norm(pca - pca2) for pca in pca1_list]))
                # 概率最大值的下标即分类标签
                if predict_scores.argmax() == row_index:
                    positive_nums += 1
                # 9 * 30, 预测标签
                predicts.extend(predict_scores)
            row_index += 1
        return np.array(target_labels), np.array(predicts), positive_nums / (class_nums * 9)

    def run(self):
        targets, predicts, accuracy = self.compute()
        print(f"PCA准确率: {round(accuracy, 2)}")
        np.save("target_labels", targets)
        np.save("pca_predicts", predicts)
        self.plot(targets, predicts, "pca_roc")


if __name__ == '__main__':
    model = PCAModel(img_nums=300)
    model.run()
