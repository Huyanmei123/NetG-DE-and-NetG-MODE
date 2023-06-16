'''
Author: your name
Date: 2020-12-27 14:21:38
LastEditTime: 2020-12-27 14:22:03
LastEditors: Please set LastEditors
Description: svm训练函数
FilePath: \DE_parallel_together\svm.py
'''
#%%
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
    auc = metrics.auc(fpr_1, tpr_1)
    time2 = time.time()
    svm_time = time2 - time1
    # print("(obj2=%.2f, tpr=recall=%.2f, fpr=%.2f, precission=%.2f, AUC=%.2f)" % (pre_tes_score, tpr, fpr, precission, AUC))
    return [tpr, fpr, precission, auc], pre_tes_score, svm_time


if __name__ == "__main__":
    dpath = './data/'
    trpath = ['alizadeh_trainData.csv', 'armstrong_trainData.csv', 'chen_trainData.csv', 'chowdary_trainData.csv', 'gordon_trainData.csv', 'nutt_trainData.csv', 'pomeroy_trainData.csv', 'shipp_trainData.csv', 'singh_trainData.csv', 'yeoh_trainData.csv', 'creditcard_trainData.csv']
    tepath = ['alizadeh_testData.csv', 'armstrong_testData.csv', 'chen_testData.csv', 'chowdary_testData.csv', 'gordon_testData.csv', 'nutt_testData.csv', 'pomeroy_testData.csv', 'shipp_testData.csv', 'singh_testData.csv', 'yeoh_testData.csv', 'creditcard_trainData.csv']
    res_dict = {'Dataset':[], 'Accuracy': [], 'Dim':[]}
    for i in range(len(trpath)):
        print(trpath[i], tepath[i])
        data_train = pd.read_csv(dpath+trpath[i], header=None, index_col=None).values
        data_test = pd.read_csv(dpath+tepath[i], header=None, index_col=None).values
        pa_svm, pre_tes_score, svm_time = svm_train_2(data_train[:, :-1], data_train[:, -1], data_test[:, :-1], data_test[:, -1])
        res_dict['Dataset'].append(trpath[i].split('_')[0])
        res_dict['Accuracy'].append(pre_tes_score)
        res_dict['Dim'].append(len(data_train[0]))
    df = pd.DataFrame(res_dict)
    df.to_csv('./result/Accuracy_Allfeatures.csv')
    # dim = 1095
    # path_train = './data/alizadeh_trainData.csv'
    # path_test = './data/alizadeh_testData.csv'
    # data_train = pd.read_csv(path_train, header=None, index_col=None).values
    # data_test = pd.read_csv(path_test, header=None, index_col=None).values
    # feats = [1055,1056,1058,1060,1065,1067,1069,1070,1071,1072,1073,1074,1076,1077,1078,1080,1081,1082,1084,1085]
    # feats = [7, 12, 36, 73, 97, 114, 119, 144, 151, 152, 157, 168, 172, 183, 202, 253, 268, 269, 273, 282, 299, 301, 343, 355, 368, 375, 383, 388, 413, 433, 441, 446, 453, 472, 488, 503, 526, 529, 539, 548, 561, 566, 567, 582, 591, 595, 599, 606, 619, 623, 649, 653, 661, 665, 671, 679, 729, 731, 737, 747, 772, 793, 806, 818, 837, 870, 879, 884, 909, 985, 995, 997, 1006, 1009, 1017, 1035, 1056, 1058, 1070, 1091]
    # feats = [16, 36, 58, 66, 71, 83, 85, 88, 122, 141, 166, 197, 223, 233, 280, 340, 363, 429, 434, 461, 472, 489, 512, 514, 519, 526, 552, 594, 623, 633, 644, 647, 651, 653, 663, 693, 696, 698, 702, 710, 723, 724, 727, 731, 742, 748, 755, 773, 774, 778, 797, 815, 860, 882, 900, 904, 933, 966, 982, 1020, 1021, 1038, 1044, 1055, 1083, 1094]
    # feats = [14, 22, 24, 27, 34, 92, 95, 119, 134, 152, 179, 182, 191, 224, 226, 227, 232, 240, 248, 253, 258, 280, 287, 288, 318, 328, 347, 363, 365, 369, 386, 393, 406, 410, 437, 459, 469, 478, 502, 513, 515, 537, 552, 554, 592, 655, 657, 659, 715, 716, 722, 742, 746, 749, 763, 791, 816, 831, 838, 885, 888, 893, 897, 921, 943, 954, 957, 960, 969, 972, 974, 994, 1008, 1025, 1044, 1053, 1068, 1076, 1081, 1084, 1088]
    # feats = [25, 29, 35, 60, 65, 66, 69, 73, 75, 77, 81, 88, 93, 94, 98, 99, 103, 111, 122, 128, 144, 148, 149, 159, 169, 173, 174, 175, 176, 180, 185, 188, 189, 192, 194, 203, 210, 211, 216, 224, 239, 241, 247, 258, 259, 263, 264, 268, 273, 274, 286, 290, 294, 300, 305, 307, 308, 311, 317, 319, 324, 330, 337, 349, 351, 359, 362, 369, 376, 381, 382, 383, 385, 387, 391, 394, 404, 406, 416, 420, 421, 428, 430, 437, 438, 453, 461, 467, 469, 475, 479, 485, 495, 500, 501, 514, 517, 527, 529, 530, 534, 535, 548, 553, 555, 572, 574, 578, 581, 594, 608, 611, 613, 621, 622, 623, 631, 635, 640, 649, 655, 661, 664, 666, 668, 681, 682, 685, 687, 710, 720, 723, 728, 730, 737, 747, 752, 755, 768, 799, 800, 805, 810, 833, 836, 840, 847, 852, 862, 875, 881, 883, 885, 892, 904, 905, 906, 909, 918, 926, 935, 951, 955, 972, 994, 1002, 1014, 1018, 1032, 1034, 1042, 1043, 1051, 1060, 1063, 1064, 1071, 1077, 1084]
    # print(len(feats))

    # pa_svm, pre_tes_score, svm_time = svm_train_2(data_train[:, feats], data_train[:, -1], data_test[:, feats], data_test[:, -1])
    # pa_svm, pre_tes_score, svm_time = svm_train_2(data_train[:, :-1], data_train[:, -1], data_test[:, :-1], data_test[:, -1])
    # print('classifier accuracy: ', pre_tes_score, '[tpr, fpr, precission, auc]: ', pa_svm, 'svm_time: ', svm_time)

    # for i in range(10):
    #     ll = np.random.randint(0, 2, dim)
    #     feats = np.where(ll == 1)[0]
    #     print("第%s次循环: " % i)
    #     # svm_train_2(train_features, train_labels, test_features, test_labels)
    #     pa_svm, pre_tes_score, svm_time = svm_train_2(data_train[:, feats], data_train[:, -1], data_test[:, feats], data_test[:, -1])
    #     print('classifier accuracy: ', pre_tes_score, '[tpr, fpr, precission, auc]: ', pa_svm, 'svm_time: ', svm_time)

#%%
# import os

# filelist = os.listdir('./data/')
# for fl in filelist:
#     print(fl)