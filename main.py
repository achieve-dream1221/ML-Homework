#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 18:19
# @Author  : achieve_dream
# @File    : main.py
# @Software: Pycharm
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题


def plot():
    """
    绘制图形
    :return: None
    """
    target_labels = np.load("runs/target_labels.npy")
    lbp = np.load("runs/lbp_predicts.npy")
    pca = np.load("runs/pca_predicts.npy")
    haar = np.load("runs/wave_predicts.npy")
    gabor = np.load("runs/gabor_predicts.npy")
    predicts = zip([lbp, pca, haar, gabor], ['LBP', "PCA", "HAAR", "GABOR"])
    for predict, name in predicts:
        fpr, tpr, _ = roc_curve(target_labels, predict)
        # roc面积
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} ROC 曲线 (面积 = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC 曲线')
    plt.legend()
    plt.savefig('runs/ROC.svg')
    plt.show()


if __name__ == '__main__':
    plot()
