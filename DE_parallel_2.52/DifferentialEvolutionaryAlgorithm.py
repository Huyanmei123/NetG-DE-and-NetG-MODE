'''
Author: lumin
Date: 2020-12-27 14:19:43
LastEditTime: 2021-11-14 23:09:10
LastEditors: Please set LastEditors
Description: DE算法函数
FilePath: \DE_parallel_together_t_v11\DifferentialEvolutionaryAlgorithm.py
'''
from os import write
import numpy as np
import pandas as pd
import csv
import math
import random
import time
from datetime import datetime
import sys

from svm import svm_train_2
import Graph
import community

from mpi4py import MPI

# %% 

class DE:
    def __init__(self, size, dim, Gm, x_min, x_max, F, CR, m_min):
        self.size = size
        self.dim = dim
        self.Gm = Gm
        self.x_min = x_min
        self.x_max = x_max
        self.F, self.F0 = F, F
        self.CR = CR
        self.m_min = m_min
        self.trainData = read_raw_data(sys.argv[1], 0)
        self.testData = read_raw_data(sys.argv[2], 0)
        self.ovs = np.zeros((size, 7), dtype=float)  # 一次迭代中，每个个体的7个metrics
        self.best_ovs = np.zeros(7, dtype=float)  # 最优解对应的7个metrics
        self.best_pos = np.zeros(dim, dtype=float)  # 最优个体的值
        self.start, self.end = self.obtain_task()
        # 记录一次迭代中，svm和pj所消耗的时间
        self.svm_time = 0.0  
        self.pj_time = 0.0
        self.create_pos()   
     
    def create_pos(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        # 主进程产生pos随机值，子进程接收主进程广播的pos值，以达到所有进程的pos初始值都一样的目的
        if rank == 0:
            np.random.seed(int(time.time()))
            pos_ini = np.random.random((self.size, self.dim))
        self.__pos = comm.bcast(pos_ini if rank == 0 else None, root=0)  # 产生初始种群
        # 更新全局最优解及对应的ovs
        start, end = self.start, self.end
        self.ovs[start:end] = np.array([self.obj_fun(self.__pos[i]) for i in range(start, end)])
        maxindex = np.argmax(self.ovs[start: end, -1])+start  # [start:end]中的最优个体
        temp_result = np.append(maxindex, self.ovs[maxindex])
        gather_result = comm.gather(temp_result, root=0)
        if rank == 0:
            gather_result = np.array(gather_result)
            index = np.argmax(gather_result[:, -1])
            result = gather_result[index]
        result = comm.bcast(result if rank == 0 else None, root=0)
        maxindex = int(result[0])
        self.best_ovs = result[1:].copy()
        self.best_pos = self.__pos[maxindex].copy()
        self.svm_time = 0.0

    def obtain_task(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nprocs = comm.Get_size()
        # main process scatter task to every process
        interval = self.size // nprocs
        remainder = self.size % nprocs
        _len = [interval+1 if i<=remainder else interval for i in range(48)]
        _len[0]-=1
        task = [[0, _len[0]]]
        for i in range(1, 48):
            start = np.sum(_len[:i])
            end = np.sum(_len[:i+1])
            task.append([start, end])
        start = task[rank][0]
        end = task[rank][1]
        return start, end

    def gather_update(self, start, end, comm, rank):
        temp_result = [self.ovs[start: end], self.__pos[start:end], start, end]
        rank0_recv = comm.gather(temp_result, root=0)  # rank0 gather results of all process in one iteration
        temp_result = []
        if rank == 0:
            # update pos
            for i in range(len(rank0_recv)):
                starti = rank0_recv[i][-2]
                endi = rank0_recv[i][-1]
                self.ovs[starti: endi] = rank0_recv[i][0].copy()
                self.__pos[starti: endi] = rank0_recv[i][1].copy()
            temp_result = [self.ovs, self.__pos]
        # send the updated pos to sub_processes
        temp_result = comm.bcast(temp_result if rank == 0 else None, root=0)
        self.ovs = temp_result[0].copy()
        self.__pos = temp_result[-1].copy()
        maxIndex = np.argmax(self.ovs[:, -1])
        self.best_ovs = self.ovs[maxIndex].copy()
        self.best_pos = self.__pos[maxIndex].copy()

    def analysis_parameters(self, flag1, flag2): 
        if flag1 == 'None' and flag2 == 'None':
            flagt = 'no'  # network
            flagg = 'no'  # community
        elif flag1 =='None' and flag2 != 'None':
            flagt = 'no'
            flagg = 'yes'
        elif flag1 != 'None' and flag2 == 'None':
            flagt = 'yes'
            flagg = 'no'
        elif flag1 != 'None' and flag2 != 'None':
            flagt = 'yes'
            flagg = 'yes'
        return flagt, flagg
        
    def get_pos(self, ith):
        return self.__pos[ith]

    # changeF, avoid lost in local best    
    def changeF(self, t):
        Gm = self.Gm
        Lambda = math.exp(1-Gm/(1+Gm-t))
        self.F = self.F0*np.exp2(Lambda)

    # 归一化
    def normalize(self, data):
        pmin = np.min(data)
        pmax = np.max(data)
        _range = pmax - pmin
        data = (data - pmin)/_range
        return data  #

    # 解码
    def decode_fun(self, _x):
        # _x = _x.copy()  # 传参传的是地址，若不复制，则会改变self.__pos的值
        x = _x.copy()
        x[x < self.m_min] = 0
        x[x >= self.m_min] = 1
        return x

    def get_feature_subset(self, pos):
        decode_pos_inv = self.decode_fun(pos)
        feats = np.where(decode_pos_inv == 1)[0]  # tuple类型，取出其中的数组值
        return feats

    # !目标函数obj_fun
    def obj_fun(self, _pos_inv):  # _pos_inv 个体的一组元素值，_X DE算法的所有pos值
        f1, f2, fitness = 0.0, 0.0, 0.0
        metrics = []
        feats = []
        Alpha = 0.3
        Beta = 0.7
        feats = self.get_feature_subset(_pos_inv)
        # 特征占比m1=子集选择特征个数/总特征个数
        # 个体准确率m2=第i个个体中包含1的个数/子集总特征个数
        count = len(feats)
        f1 = -count/self.dim
        if count > 0:
            # 从原始数据中取出相应列，进行训练
            train_features = self.trainData[:, feats]
            train_labels = self.trainData[:, -1]
            test_features = self.testData[:, feats]
            test_labels = self.testData[:, -1]
            metrics, f2, svm_time = svm_train_2(train_features, train_labels, test_features, test_labels)
            self.svm_time += svm_time
            fitness = Alpha * f1 + Beta * f2
        metrics = np.append(metrics, [f1,f2,fitness])
        return metrics  # 返回目标概率

    def execution(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        h1 = ['Tpr', 'Fpr', 'Precission', 'Auc', 'Ratio', 'Accuracy', 'Fitness']
        h2 = ['svm_t', 'pj_t', 'de_t', 'comm_t']  # 各进程通信的时间
        timestamp = time.strftime('%Y-%m-%d %H%M%S', time.localtime())
        g = None
        c = None
        title = ''
        time_complexity, ovs_list, best_pos_list = [], [], []
        weight = float(sys.argv[6])  # 如果选择的是没有网络结构的DE算法，则此参数表示CR，否则表示weight
        dataset = sys.argv[11]
        start, end = self.start, self.end
        flag1 = sys.argv[3]  # network
        flag2 = sys.argv[4]  # community
        flagt, flagg=self.analysis_parameters(flag1, flag2)
        if flagt == 'yes' and flagg == 'yes':  # 有网络结构有社区结构
            # main process load graph and community, then broadcast to others
            if rank == 0:
                title = dataset + ' NetG-DE weight_' + str(weight) + ' iterNum_' + str(self.Gm) + ' ' + timestamp
                g = Graph.Graph(flag1, weight)  # 加载网络结构
                c = community.CommunityGroup(flag2)  # 加载社区网络分组
                print('runing NetG-DE, weight=%.2f!' % (weight))
            g = comm.bcast(g if rank == 0 else None, root=0)
            c = comm.bcast(c if rank == 0 else None, root=0)
        elif flagt == 'no' and flagg == 'no' and rank == 0:  # 无社区结构无网络结构
            title = dataset + ' DE CR_' + str(self.CR) + ' iterNum_' + str(self.Gm) + ' ' + timestamp
            print('runing DE, CR=%.2f!' % (weight))
        elif flagt == 'yes' and flagg == 'no':  # 没有社区结构，只有网络结构
            if rank == 0:
                title = dataset + ' Net-DE weight_' + str(weight) + ' iterNum_' + str(self.Gm) + ' ' + timestamp
                g = Graph.Graph(flag1, weight)  # 加载网络结构
                print('runing Net-DE, weight=%.2f!' % (weight))
            g = comm.bcast(g if rank == 0 else None, root=0)
        elif flagt == 'no' and flagg == 'yes':  # 没有网络结构，只有社区结构
            if rank == 0:
                title = dataset + ' Com-DE weight_' + str(weight) + ' iterNum_' + str(self.Gm) + ' ' + timestamp
                g = Graph.Graph(flag1, weight)  # 加载网络结构
                c = community.CommunityGroup(flag2)  # 加载社区网络分组
                print('runing Com-DE, weight=%.2f!' % (weight))
            g = comm.bcast(g if rank == 0 else None, root=0)
            c = comm.bcast(c if rank == 0 else None, root=0)
        if rank==0:
            path1 = './result_12/excels/excels ' + title + '.csv'
            path2 = './result_12/featureSubsets/FS '+title+'.csv'
            path3 = './result_12/timeEstimation/time ' + title + '.csv'
        
        for t in range(1, self.Gm+1):
            self.svm_time, self.pj_time = 0.0, 0.0  # 时间数据置零
            time1 = time.time()
            for i in list(range(start, end)):
                u = self.mutation(i)  # 变异
                if flagt == 'no' and flagg == 'no':  # 如果选择的是无社区结构的DE算法，则选择CR作为参考
                    v = self.crossover_noNetnoCom(self.__pos[i], u)  # 交叉
                elif flagt == 'yes' and flagg=='yes':  # 如果选择的是有社区结构的，则选择pj作为参考
                    v = self.crossover_NetG(self.__pos[i], u, g, c)  # 交叉
                elif flagt == 'no' and flagg=='yes':
                    v = self.crossover_Community(self.__pos[i], u, g, c)
                elif flagt == 'yes' and flagg == 'no':
                    v = self.crossover_Network(self.__pos[i], u, g)  # crossover
                self.selection(i, v)  # 选择,从原始个体和交叉个体中选择；已经更新了下一代的个体元素值，种群best_fitness_value和best_pos(最优个体)
                time2 = time.time()
            self.gather_update(start, end, comm, rank)  # 根据各进程的数据，更新X(g+1),以及对应的ovs；更新最优解及其ovs
            if rank == 0:
                ovs_list.append(self.best_ovs)
                best_pos_list.append(self.best_pos)
                time3 = time.time()
                time_complexity.append([self.svm_time, self.pj_time, time2-time1, time3-time2])
                print('iteration', t, ',[Tpr,Fpr,Precission,Auc,Ratio,Accuracy,Fitness]=', self.best_ovs)
            self.changeF(t)
        # 将metrics, 时间数据，feature subset写入文件
        if rank == 0:
            self.write_result(ovs_list, best_pos_list, time_complexity, h1, h2, path1, path2, path3)
            print('finished %s, network: %s, community: %s, CR=%.2f, weight=%.2f' % (dataset, flagt, flagg, self.CR, weight))
       
    # 变异
    def mutation(self, i):
        j = k = 0
        # 保证i != j != k
        while j == k or j == i or k == i:
            j = random.randint(0, self.size - 1)
            k = random.randint(0, self.size - 1)
        u = self.get_pos(i) + self.F * (self.get_pos(j) - self.get_pos(k))
        # 对变异后的个体元素进行归一化
        u = self.normalize(u)
        return u

    # 无社区结构的 交叉算法
    # 交叉 选择变异基因或者原始基因
    def crossover_noNetnoCom(self, pos, u):
        v = np.zeros((self.dim,), float)
        ll = list(range(self.dim))
        random.shuffle(ll)
        v = np.array([u[j] if random.random() < self.CR else pos[j] for j in ll])
        return v
        
    # network, but no community
    # 交叉 选择变异基因或者原始基因
    def crossover_Network(self, pos, u, g):
        v = np.zeros((self.dim,), float)
        decode_u = self.decode_fun(u)
        Alpha = float(sys.argv[7])
        Beta = float(sys.argv[8])
        pj = 0.0
        ll = list(range(self.dim))
        random.shuffle(ll)
        for j in ll:
            rand_value = random.random()
            if u[j] == pos[j]:
                v[j] = pos[j]
                continue
            elif rand_value < self.CR:
                v[j] = u[j]
                continue
            time1 =time.time()
            # 第j位对应的特征节点的加权度
            djw = g.get_weight_degree(str(j))
            if djw == 0.001:
                v[j]=pos[j]
                continue
            # 否则需要网络与社区结构
            # 该特征节点的邻接点中被置为1的加权个数，在变异后的个体中选择元素，并在graph中找该特征节点与对应特征节点的weight，并求和
            wjn = g.cal_weight_number(str(j), decode_u)
            pj = Alpha * math.exp(-3/djw) + Beta * math.exp(-wjn)
            # print('weight degree=', djw, 'weight number=', wjn, 'pj=', pj)
            time2 = time.time()
            self.pj_time += (time2-time1)
            if rand_value < self.CR*(1-pj)+pj:
                maxpv = max(u[j], pos[j])
                v[j] = maxpv  # [0.5, 1), 交叉个体选择the current feature
            elif rand_value>=self.CR*(1-pj)+pj:
                minpv = min(u[j], pos[j])
                v[j] = minpv  # [0, 0.5), 交叉个体不选择该特征
        return v

    # community, but no network
    def crossover_Community(self, pos, u, g, c):
        v = np.zeros((self.dim,), float)
        decode_u = self.decode_fun(u)
        Alpha = float(sys.argv[7])
        Gamma = float(sys.argv[9])
        pj = 0.0
        ll = list(range(self.dim))
        random.shuffle(ll)
        for j in ll:
            rand_value = random.random()
            if u[j] == pos[j]:
                v[j] = pos[j]
                continue
            elif rand_value < self.CR:
                v[j] = u[j]
                continue
            time1 = time.time()
            # 第j位对应的特征节点的加权度
            djw = g.get_weight_degree(str(j))
            if djw == 0.001:
                v[j]=pos[j]
                continue
            # 该特征节点所在的特征组中被置为1的个数， 在community中找到特征组，并获取该组的特征节点，判断u中的个体元素对应位是否为1
            njc = c.cal_selected_number(str(j), decode_u)
            pj = Alpha * math.exp(-3/djw) + Gamma * math.exp(-njc)
            # print('weight degree=', djw, 'selected node number(community)=', njc, 'pj=', pj)
            time2 = time.time()
            self.pj_time += (time2-time1)
            if rand_value < self.CR*(1-pj)+pj:
                maxpv = max(u[j], pos[j])
                # if maxpv < self.m_min:
                #     maxpv += 0.5
                v[j] = maxpv  # [0.5, 1), 交叉个体选择the current feature
            elif rand_value>=self.CR*(1-pj)+pj:
                minpv = min(u[j], pos[j])
                # if minpv >= self.m_min:
                #     minpv -= 0.5
                v[j] = minpv  # [0, 0.5), 交叉个体不选择该特征
        return v

    # network+community
    # 交叉 选择变异基因或者原始基因
    def crossover_NetG(self, pos, u, g, c):
        v = np.zeros((self.dim,), float)
        decode_u = self.decode_fun(u)
        Alpha = float(sys.argv[7])
        Beta = float(sys.argv[8])
        Gamma = float(sys.argv[9])
        pj = 0.0
        ll = list(range(self.dim))
        random.shuffle(ll)
        for j in ll:
            rand_value = random.random()
            if u[j] == pos[j]:
                v[j] = pos[j]
                continue
            elif rand_value < self.CR:
                v[j] = u[j]
                continue
            time1 = time.time()
            # 第j位对应的特征节点的加权度
            djw = g.get_weight_degree(str(j))
            if djw == 0.001:
                v[j]=pos[j]
                continue
            # 该特征节点的邻接点中被置为1的加权个数，在变异后的个体中选择元素，并在graph中找该特征节点与对应特征节点的weight，并求和
            wjn = g.cal_weight_number(str(j), decode_u)
            # 该特征节点所在的特征组中被置为1的个数， 在community中找到特征组，并获取该组的特征节点，判断u中的个体元素对应位是否为1
            njc = c.cal_selected_number(str(j), decode_u)
            pj = Alpha * math.exp(-3/djw) + Beta * math.exp(-wjn) + Gamma * math.exp(-njc)
            time2 = time.time()
            self.pj_time += (time2-time1)
            # print('weight_degree=', djw, 'weight number=', wjn, 'com number=', njc, 'pj=', pj)
            if rand_value < self.CR*(1-pj)+pj:
                maxpv = max(u[j], pos[j])
                v[j] = maxpv  # [0.5, 1), 交叉个体选择the current feature
            elif rand_value>=self.CR*(1-pj)+pj:
                minpv = min(u[j], pos[j])
                v[j] = minpv  # [0, 0.5), 交叉个体不选择该特征
        return v

    # 选择  更新下一代的种群个体，全局最优解，全局最优适应值
    def selection(self, i, v):
        ovs_v = self.obj_fun(v)
        if ovs_v[-1] >= self.ovs[i][-1]:  # 选择新个体
            self.__pos[i] = v.copy()
            self.ovs[i] = ovs_v.copy()
            # 判断是否是全局最优解
            if ovs_v[-1] > self.best_ovs[-1]:
                # 更新全局最优解和最佳适应值
                self.best_pos = v.copy()
                self.best_ovs = ovs_v.copy()
        else:
            # 判断是否需要更新全局最优解和最佳适应值
            if self.ovs[i][-1] > self.best_ovs[-1]:
                self.best_pos = self.__pos[i].copy()
                self.best_ovs = self.ovs[i].copy()

    def write_result(self, ovs_list, best_pos_list, time_complexity, h1, h2, path1, path2, path3):
        write_to_csv(path1, ovs_list, h1)
        write_to_csv(path3, time_complexity, h2)
        # 将best_pos转成feature subset
        feature_subsets = [self.get_feature_subset(best_pos_list[i]) for i in range(self.Gm)]
        write_to_csv(path2, feature_subsets, None)

def read_raw_data(path, header):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # rank = 0
    time1 = time.time()
    data = pd.read_csv(path, header=header).values
    time2 = time.time()
    if rank == 0:
        print("read data from %s, takes up %s seconds" % (path, time2 - time1))
    return data

# 将data写入csv文件(矩阵or维度不一致的二维数据都可)
def write_to_csv(path, data, h):
    df = pd.DataFrame(data)
    df.to_csv(path, header=h)

