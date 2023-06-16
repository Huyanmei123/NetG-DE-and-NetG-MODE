'''
Author: your name
Date: 2020-12-27 14:21:38
LastEditTime: 2020-12-27 14:22:03
LastEditors: Please set LastEditors
Description: svm训练函数
FilePath: \DE_parallel_together\svm.py
'''

import numpy as np
import pandas as pd
import time
from sklearn import svm, metrics
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


def loadDataCsv(path):
    rawData = pd.read_csv(path, header=None)
    data = rawData.values
    return data

def svm_train_2(train_features, train_labels, test_features, test_labels):
    # 训练svm分类器
    time1 = time.time()
    clf = svm.SVC(C=10, kernel='rbf', gamma='auto', decision_function_shape='ovr')
    y_score = clf.fit(train_features, train_labels.ravel()).decision_function(test_features)  # 预测为0或1的概率
    
    # 计算svm分类器的准确率
    pre_tes_label = clf.predict(test_features)  # 测试集的预测标签
    pre_tes_score = accuracy_score(test_labels, pre_tes_label)  # 测试集的分类准确率

    # 求tpr，fpr的值
    tn, fp, fn, tp = confusion_matrix(test_labels, pre_tes_label).ravel()
    tpr = tp/(tp+fn)
    fpr = fp/(tn+fp)
    precission = tp/(tp+fp)
    # 计算roc曲线下方的面积，即auc的值
    fpr_1, tpr_1, threshold_1 = metrics.roc_curve(test_labels, y_score, pos_label=1)  # 不同阈值下，tpr和fpr的值
    AUC = metrics.auc(fpr_1, tpr_1)
    time2 = time.time()
    svm_time = time2 - time1
    # print("(obj2=%.2f, tpr=recall=%.2f, fpr=%.2f, precission=%.2f, AUC=%.2f)" % (pre_tes_score, tpr, fpr, precission, AUC))
    return [tpr, fpr, precission, AUC],pre_tes_score, svm_time


if __name__ == "__main__":
    dim = 87
    path1 = 'D:/MyFile/AProgram/workspace/the-first-topic new/data/processedData/arcene_trainData.csv'
    path2 = 'D:/MyFile/AProgram/workspace/the-first-topic new/data/processedData/arcene_testData.csv'
    
    traindata = loadDataCsv(path1)
    testdata = loadDataCsv(path2)
    for i in range(10):
        ll = np.random.randint(0, 2, dim)
        feats = np.where(ll == 1)[0]
        train_features = traindata[:, feats]  # 选择对应特征列的数据
        train_labels = traindata[:, -1]  # 标签
        test_features = testdata[:, feats]
        test_labels = testdata[:, -1]
        print("第%s次循环：" % i)
        # svm_train_2(train_features, train_labels, test_features, test_labels)
        pre_tes_score, pa_svm, svm_time = svm_train_2(train_features, train_labels, test_features, test_labels)
        print('classifier accuracy: ', pre_tes_score, '[tpr, fpr, precission, auc]: ', pa_svm, 'svm_time: ', svm_time)
