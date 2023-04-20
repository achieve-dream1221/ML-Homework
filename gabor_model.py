#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 17:13
# @Author  : achieve_dream
# @File    : gabor_model.py
# @Software: Pycharm
import cv2
import numpy as np

from model import Model


class GaborModel(Model):
    def compute(self):
        # 存储第一列的gabor, 方便后面的数据集和它进行比较
        k_size = 31  # 滤波器大小
        sigma = 5  # 高斯核标准差
        theta = np.pi / 4  # 方向
        lambd = 10  # 波长
        gamma = 0.5  # 空间纵横比
        phi = 0.5  # 相位偏移
        # 计算Gabor滤波器的卷积核
        kernel = cv2.getGaborKernel((k_size, k_size), sigma, theta, lambd, gamma, phi)
        gabor1_list = [cv2.filter2D(self.read_img(col_1), cv2.CV_32F, kernel) for col_1 in self.dataset[:, 0]]
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
                gabor2 = cv2.filter2D(self.read_img(col), cv2.CV_32F, kernel)
                # 先计算lbp的欧式距离, 然后通过softmax预测出这10个类别的概率
                predict_scores = self.softmax(np.array([np.linalg.norm(gabor - gabor2) for gabor in gabor1_list]))
                # 概率最大值的下标即分类标签
                if predict_scores.argmax() == row_index:
                    positive_nums += 1
                # 9 * 30, 预测标签
                predicts.extend(predict_scores)
            row_index += 1
        return np.array(target_labels), np.array(predicts), positive_nums / (class_nums * 9)

    def run(self):
        targets, predicts, accuracy = self.compute()
        print(f"Gabor准确率: {round(accuracy, 2)}")
        np.save("target_labels", targets)
        np.save("gabor_predicts", predicts)
        self.plot(targets, predicts, "gabor_roc")


if __name__ == '__main__':
    model = GaborModel(img_nums=300)
    model.run()
