'''
Author: 陆敏
Date: 2020-12-29 19:26:18
LastEditTime: 2021-11-13 22:46:09
LastEditors: Please set LastEditors
Description: 多目标的差分进化算法
FilePath: /MODE_parallel_together/MODE.py
'''
#%%
import numpy as np
import pandas as pd
import math
import random
import time
import sys

from svm import svm_train_2
import Graph1 as Graph
import community1 as community
from mpi4py import MPI

import utils

#%%
Dom_dir = utils.Dom_cr_dir
timeEstimation_dir = utils.timeEstimation_cr_dir

# %%
# load data
# train_path = '../data/shipp_trainData.csv'
# test_path = '../data/shipp_testData.csv'
train_path = sys.argv[1]
test_path = sys.argv[2]
train_data = pd.read_csv(train_path, header=None).values
test_data = pd.read_csv(test_path, header=None).values

#%%
def write_rows(path, data, h):
    df = pd.DataFrame(data)
    df.to_csv(path, header=h)

#%%

class MODE:
    def __init__(self, N, dim, Gm, M, F, CR, m_min=0.5):
        self.N = N
        self.dim = dim
        self.Gm = Gm
        self.M = M  # M-objectives problems
        self.F0,self.F = F, F
        self.CR = CR
        self.m_min = m_min
        self.X = np.random.random((3*N, dim)) # parents, mutation, crossover
        self.X_dec = np.zeros((3*N, dim), dtype=int)
        self.bins = int(self.N * 0.6)
        # self.U = np.zeros((self.N, self.dim), dtype=float)  # save mutation individual
        # self.V = np.zeros((self.N, self.dim), dtype=float) # save crossover individual
        self.Dm = -np.ones((self.M, self.bins+1), dtype=int)  # sign the nondominate individuals
        self.Ovs = np.zeros((3*N, 6), dtype=float)  # 每个个体(X and trial)的m个目标函数值
        self.Fmin = np.zeros(M, dtype=float)
        self.Fmax = np.zeros(M, dtype=float)
        self.Dom = np.array([])  # 最优解的7个评价指标 
        self.best_pos = np.array([])  # 最优解的个体值
        self.timeCost = np.zeros((Gm+1, 3), dtype=float)  # Total time, Search time, svm time
        [self.obj_fun(i, 0) for i in range(0, N)]  # 更新目标函数值以及Dom解集

    def get_pos(self, ith):
        return self.X[ith]
    
    # def changeF(self, t):
    #     Lambda = math.exp(1-self.Gm/(1+self.Gm-t))
    #     self.F = self.F0 * np.exp2(Lambda)

    # normalize
    def normalize(self, data):
        pmin = np.min(data)
        pmax = np.max(data)
        _range = pmax - pmin
        return (data - pmin)/_range
    
    # decode
    def decode(self, _x):
        dec_x = _x.copy()
        dec_x[dec_x < self.m_min] = 0
        dec_x[dec_x >= self.m_min] = 1
        return dec_x

    def obj_fun(self, ith, t):
        f1, f2 = 0.0, 0.0
        # feats, metrics = [], []
        metrics = -np.ones(6, dtype=float)
        # if ith % 3 == 0:
        #     cur_indv = self.X[ith//3].copy()
        # elif ith%3==1:
        #     cur_indv = self.V[ith//3].copy()
        # elif ith%3==2:
        #     cur_indv = self.U[ith//3].copy()
        self.X_dec[ith] = self.decode(self.X[ith])
        feats = np.where(self.X_dec[ith] == 1)[0]  # tuple类型，取出其中的数组值
        count = len(feats) 
        # f1=个体中为1的元素个数/dim, 特征占比
        f1 = - count/self.dim
        if count > 0:
            # 从原始数据中取出相应的列进行训练
            train_features, train_labels = train_data[:, feats], train_data[:, -1]
            test_features, test_labels = test_data[:, feats], test_data[:, -1]
            self.Ovs[ith, :4], f2, svm_t = svm_train_2(train_features, train_labels, test_features, test_labels)
            self.timeCost[t, -1] += svm_t
            self.Ovs[ith, -2:] = [f1, f2]
            # self.Ovs[ith] = metrics
            self.updateDominateset(ith)

    def write_Dom(self, t, path, h):
        tt = [t] * len(self.Dom)
        temp_dom = np.c_[tt, self.Dom]
        # write_rows_toCsv(path, temp_dom, h)
        write_rows(path, temp_dom, h)
        # return temp_dom

    # 变异                
    def mutation(self, i, t):
        j = k = 0
        while j == i:
            j = random.randint(0, self.N - 1)
        while k == i or k == j:
            k = random.randint(0, self.N - 1)
        v = self.get_pos(i) + self.F * (self.get_pos(j) - self.get_pos(k))
        newindex = i+self.N
        self.X[newindex] = self.normalize(v)
        self.obj_fun(newindex, t)

# 无网络结构、无社区结构的交叉, 即标准DE的 crossover操作
    def crossover_canonical(self, i, t):
        ll = list(range(self.dim))
        random.shuffle(ll)
        mindex = i+self.N
        cindex = i+2*self.N
        v = self.X[mindex]
        pos = self.X[i]
        self.X[cindex] = np.array([v[j] if random.random() < self.CR else pos[j] for j in ll])
        self.obj_fun(cindex, t)
    

# 有网络结构，无社区结构的交叉
    def crossover_Network(self, i, t, g, Alpha, Beta, vers):
        mindex = i+self.N
        cindex = i+2*self.N
        u = np.zeros((self.dim), float)
        pj = 0.0
        ll = list(range(self.dim))
        random.shuffle(ll)
        v = self.X[mindex]
        decode_v = self.X_dec[mindex]
        pos = self.X[i]
        for j in ll:
            rand_value = random.random()
            if rand_value < self.CR:
                u[j] = v[j]
                continue
            elif v[j] == pos[j] or vers[j]==0:
                u[j] = pos[j]
                continue
            # time1 = time.time()
            # 第j位对应的特征节点的加权度
            djw = g.get_weight_degree(str(j))
            # 该特征节点的邻接点中被置为1的加权个数，在变异后的个体中选择元素，并在graph中找到该特征节点与对应特征节点的weight，并求和
            wjn = g.cal_weight_number(str(j), decode_v)
            pj = Alpha * math.exp(-3/djw) + Beta * math.exp(-wjn)
            # time2 = time.time()
            # self.pj_time += (time2-time1)
            if rand_value < self.CR*(1-pj)+pj:
                maxpv = max(v[j], pos[j])
                u[j] = maxpv  # [0.5, 1), 交叉个体选择the current feature
            elif rand_value>=self.CR*(1-pj)+pj:
                minpv = min(v[j], pos[j])
                u[j] = minpv  # [0, 0.5), 交叉个体不选择该特征
        self.X[cindex] = u
        self.obj_fun(cindex, t)

# 有网络结构 有社区结构的交叉
    def crossover_NetG(self, i, g, c, t, Alpha, Beta, Gamma, vers):
        mindex = i+self.N
        cindex = i+2*self.N
        u = np.zeros((self.dim), float)
        pj = 0.0
        ll = list(range(self.dim))
        random.shuffle(ll)
        v = self.X[mindex]  # 赋值地址
        decode_v = self.X_dec[mindex]
        pos = self.X[i]
        for j in ll:
            rand_value = random.random()
            if rand_value < self.CR:
                u[j] = v[j]
                continue
            elif v[j] == pos[j] or vers[j]==0:
                u[j] = pos[j]
                continue
            # time1 = time.time()
            # 第j位对应的特征节点的加权度
            djw = g.get_weight_degree(str(j))
            # 该特征节点的邻接点中被置为1的加权个数，在变异后的个体中选择元素，并在graph中找该特征节点与对应特征节点的weight，并求和
            wjn = g.cal_weight_number(str(j), decode_v)
            # 该特征节点所在的特征组中被置为1的个数， 在community中找到特征组，并获取该组的特征节点，判断u中的个体元素对应位是否为1
            if vers[j]==2:
                # 该特征节点所在的特征组中被置为1的个数， 在community中找到特征组，并获取该组的特征节点，判断u中的个体元素对应位是否为1
                njc = c.cal_selected_number(str(j), decode_v)
                pj = Alpha * math.exp(-3/djw) + Beta * math.exp(-wjn) + Gamma * math.exp(-njc)
            else:
                pj = 0.5*math.exp(-3/djw)+0.5*math.exp(-wjn)
            # time2 = time.time()
            # self.pj_time += (time2-time1)
            if rand_value < self.CR*(1-pj)+pj:
                maxpv = max(v[j], pos[j])
                u[j] = maxpv  # [0.5, 1), 交叉个体选择the current feature
            elif rand_value>=self.CR*(1-pj)+pj:
                minpv = min(v[j], pos[j])
                u[j] = minpv  # [0, 0.5), 交叉个体不选择该特征
        self.X[cindex] = u
        self.obj_fun(cindex, t)

    def select(self):
        self.Dm = -np.ones((self.M, self.bins+1), dtype=int)
        # 找到self.Ovs中的每个目标的最大最小值
        self.Fmin, self.Fmax = np.min(self.Ovs[:, -2:], axis=0), np.max(self.Ovs[:, -2:], axis=0)
        preindex = np.array([], dtype=int)
        selectIndex = np.array([], dtype=int)
        backup = np.array([], dtype=int)      
        # 将值分配到bins个块里面
        for m in range(-self.M, 0):
            unit = (self.Fmax[m] - self.Fmin[m]) / self.bins
            if unit == 0:
                continue
            for i in range(self.N):
                k = (self.Ovs[i, m] - self.Fmin[m]) / unit
                k = int(k)
                if self.Dm[m, k] == -1:
                    self.Dm[m, k] = i
                elif self.Ovs[i, m] > self.Ovs[self.Dm[m, k], m]:
                    self.Dm[m, k] = i
                
                mindex = i+self.N
                k = (self.Ovs[mindex, m] - self.Fmin[m]) / unit
                k = int(k)
                if self.Dm[m, k] == -1:
                    self.Dm[m, k] = mindex
                elif self.Ovs[mindex, m] > self.Ovs[self.Dm[m, k], m]:
                    self.Dm[m, k] = mindex

                cindex = i+2*self.N
                k = (self.Ovs[cindex, m] - self.Fmin[m]) / unit
                k = int(k)
                if self.Dm[m, k] == -1:
                    self.Dm[m, k] = cindex
                elif self.Ovs[cindex, m] > self.Ovs[self.Dm[m, k], m]:
                    self.Dm[m, k] = cindex

        # 取出Dm中!=-1的值
        for m in range(self.M):
            preindex = np.append(preindex, self.Dm[m, self.Dm[m] > -1])
        preindex = set(preindex)  # 去重
        if len(preindex) >= self.N:
            selectIndex = np.random.choice(list(preindex), self.N, replace=False)
        else:
            ll = set(np.arange(0, 3*self.N))
            backup = list(ll.symmetric_difference(preindex))
            surplus = self.N - len(preindex)
            surplusIndex = np.random.choice(backup, surplus, replace=False)
            selectIndex = np.append(list(preindex), surplusIndex)
        
        # update the parents and obj values
        self.X[:self.N] = self.X[selectIndex].copy()
        self.Ovs[:self.N] = self.Ovs[selectIndex].copy()
        

    # 更新最优解集
    def updateDominateset(self, ith):
        flagg = False  # 判断是否可以支配Dom中的解
        flagt = False  # 判断是否为非支配解
        if len(self.Dom) == 0:
            self.Dom = np.array([self.Ovs[ith].copy()])
            self.best_pos = np.array([self.X[ith].copy()])
            return 0 
        delIndex = []
        for d in range(len(self.Dom)):
            flag1 = False  # 判断是否有比它大的
            flag2 = False  # 判断是否有比它小的
            for m in range(-self.M, 0):
                if self.Ovs[ith, m] > self.Dom[d, m]:
                    flag1 = True
                elif self.Ovs[ith, m] < self.Dom[d, m]:
                    flag2 = True
            if flag1 is True and flag2 is True:
                flagt = True  # 表示该indv是nondominate solution
            elif flag1 is True and flag2 is False:
                flagg = True  # 表示该indiv可支配原有解集中的解d，那么应删除原解集中的对应解
                delIndex.append(d)
                flagt = False
            elif flag1 is False:
                # 表明它可以被原解集中的解支配，那么它就不应该加入到解集中
                flagt = False
                flagg = False
                break
        if len(delIndex) > 0:
            self.Dom = np.delete(self.Dom, delIndex, 0)
            self.best_pos = np.delete(self.best_pos, delIndex, 0)
        # 如果该解为非支配解(或者可以支配其他解)，则应该加入到解集中
        if flagg is True or flagt is True:
            self.Dom = np.concatenate((self.Dom, [self.Ovs[ith].copy()]), 0)
            self.best_pos = np.concatenate((self.best_pos, [self.X[ith].copy()]), 0)
      

    def execution(self, dataset):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        algorithm = ''
        g, c = None, None
        path_network = sys.argv[3]
        path_com = sys.argv[4]
        weight = float(sys.argv[5])
        Alpha, Beta, Gamma = float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8])
        flag = sys.argv[11] # 3代表NetG-GA, 2代表Net-GA, 1代表GA

        # path_network = '../data/shipp-2002-v1-norm-pearson06.txt'
        # path_com = '../data/shipp-norm-com0.75.txt'
        # weight= 0.75
        # Alpha, Beta, Gamma = 0.3, 0.3, 0.4
        # # Alpha, Beta, Gamma = 0.5, 0.5, 0.0  
        # flag = '3'

        if flag=='1':
            algorithm = 'MODE'
            print('rank %d running %s for %s' % (rank, algorithm, dataset))
        elif flag=='2':
            algorithm='Net-MODE'
            g = Graph.Graph(path_network, weight)
            ver_network = list(g.getVertices())
            vers = np.zeros(self.dim, dtype=int)
            vers[ver_network] = 1
            print('rank %d running %s for %s, weight=%.2f' % (rank, algorithm, dataset, weight))
        else:
            algorithm = 'NetG-MODE'
            g = Graph.Graph(path_network, weight)
            c = community.CommunityGroup(path_com)
            ver_network = list(g.getVertices())  # 网络结构中的节点
            ver_com = list(c.getAllVertices())  # 社区中的节点
            vers = np.zeros(self.dim, dtype=int)
            vers[ver_network] = 1
            vers[ver_com] = 2
            print('rank %d running %s for %s, weight=%.2f' % (rank, algorithm, dataset, weight))
        
        timestamp = time.strftime("%Y-%m-%d %H%M%S", time.localtime())
        str1 = '{} {} weight_{} Iteration_{} N_{} CR_{} rank_{} {}'.format(dataset, algorithm, weight, self.Gm, self.N, self.CR, rank, timestamp)
        # str1 = dataset +' ' + algorithm + ' weight_' + str(weight) + ' Iteration_' + str(self.Gm) + ' N_' + str(self.N) + ' rank_' + str(rank) + ' ' + timestamp
        # fit_path = excels_dir + 'excels ' + str1 + '.csv'
        Dom_path = Dom_dir + 'Dom ' + str1 + '.csv'
        time_path = timeEstimation_dir + 'timeEstimation ' + str1 + '.csv'
        # fit_h = ['Tpr', 'Fpr', 'Precission', 'Auc', 'Ratio', 'Accuracy', 'Fitness']
        dom_h = ['Tpr', 'Fpr', 'Precission', 'Auc', 'Ratio', 'Accuracy']
        time_h = ['Total_t', 'Search_t', 'Svm_t']

        for t in range(1, self.Gm+1):
            time1 = time.time()
            for i in range(self.N):
                self.mutation(i, t) # mutation population: V
                if flag == '1':
                    self.crossover_canonical(i, t)
                elif flag == '2':
                    self.crossover_Network(i, t, g, Alpha, Beta, vers)
                elif flag == '3':
                    self.crossover_NetG(i, g, c, t, Alpha, Beta, Gamma, vers)
            self.select()
            time2 = time.time()
            total_t = time2 - time1
            self.timeCost[t, :2] = [total_t, total_t - self.timeCost[t, -1]]
            # self.changeF(t)
            # self.svm_time, self.pj_time = 0.0, 0.0
            print("rank {}, finished iteration {}.".format(rank, t))
        # self.write_Dom(t, Dom_path, dom_h)  # 主进程把每次迭代的最优解写入文件
        write_rows(Dom_path, self.Dom, dom_h)
        write_rows(time_path, self.timeCost, time_h)
        print("rank {} finished {} for {}!".format(rank, algorithm, dataset))


#%%
if __name__ == "__main__":
    np.random.seed(int(np.random.rand()*100000))
    dim = len(train_data[0])-1
    dataset = sys.argv[10]
    iteration = int(sys.argv[9])
    N=int(sys.argv[12])
    # dataset = 'Shipp'
    # iteration = 100
    # N = 200
    m_min = 0.5
    F, CR = 0.6, float(sys.argv[13])
    # F, CR = 0.5, 0.1
    M=2
    de = MODE(N, dim, iteration, M, F, CR, m_min)
    de.execution(dataset)

