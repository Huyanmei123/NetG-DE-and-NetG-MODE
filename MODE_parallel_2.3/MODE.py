'''
Author: 陆敏
Date: 2020-12-29 19:26:18
LastEditTime: 2021-11-13 22:46:09
LastEditors: Please set LastEditors
Description: 多目标的差分进化算法
FilePath: /MODE_parallel_together/MODE.py
'''
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


class DE:
    def __init__(self, size, dim, Gm, M, F, CR, m_min):
        self.size = size
        self.dim = dim
        self.Gm = Gm
        self.M = M  # M-objectives problems
        self.F0,self.F = F, F
        self.CR = CR
        self.m_min = m_min
        self.trainData = read_raw_data(sys.argv[1])
        self.testData = read_raw_data(sys.argv[2])
        self.bins = int(self.size * 0.6)
        self.U = np.zeros((self.size, self.dim), dtype=float)  # save mutation individual
        self.V = np.zeros((self.size, self.dim), dtype=float) # save crossover individual
        self.Dm = -np.ones((self.M, self.bins+1), dtype=int)  # sign the nondominate individuals
        self.Ovs = np.zeros((3*self.size, self.M), dtype=float)  # 每个个体(X and trial)的m个目标函数值
        self.Fmin = np.zeros(self.M, dtype=float)
        self.Fmax = np.zeros(self.M, dtype=float)
        self.Dom = np.array([])  # 最优解的7个评价指标 
        self.best_pos = np.array([])  # 最优解的个体值
        self.start, self.end = self.obtain_task()
        self.svm_time = 0.0
        self.pj_time = 0.0
        self.create_pos()

    def create_pos(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            np.random.seed(int(time.time()))
            pos_ini = np.random.random((self.size, self.dim))
        self.__pos = comm.bcast(pos_ini if rank == 0 else None, root=0)
        # 更新Dom,所有进程的Dom必须保持一致
        [self.obj_fun(3*i) for i in range(self.start, self.end)]  # 更新目标函数值以及Dom解集
        # print('rank ', rank, 'dom=', self.Dom)
        gather_results = comm.gather([self.Dom, self.best_pos], root=0)
        if rank == 0:
            for i in range(len(gather_results)):
                new_dom = gather_results[i][0].copy()
                new_pos = gather_results[i][1].copy()
                self.MergeDominateset(new_dom, new_pos)
            print('finished initialization!')
        temp_data = [self.Dom, self.best_pos]
        temp_data = comm.bcast(temp_data if rank==0 else None, root=0)
        self.Dom = temp_data[0].copy()
        self.best_pos = temp_data[1].copy()

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

    def gather_result(self, comm, rank, t):
        select_t=0.0
        time1 = time.time()
        start, end = self.start, self.end
        tempdata = [start, end, self.U[start:end], self.V[start:end], self.Ovs[3*start:3*end], self.Dom, self.best_pos]
        receive_result = comm.gather(tempdata, root=0)
        # 由主进程进行选择操作
        if rank == 0:
            for i in range(len(receive_result)):
                start1 = receive_result[i][0]
                end1 = receive_result[i][1]
                self.U[start1: end1] = receive_result[i][2].copy()
                self.V[start1:end1] = receive_result[i][3].copy()
                self.Ovs[3*start1:3*end1] = receive_result[i][4].copy()
                new_dom = receive_result[i][-2].copy()
                new_pos = receive_result[i][-1].copy()
                # update Dom
                self.MergeDominateset(new_dom, new_pos)
            time3 = time.time()
            self.select()
            time4 = time.time()
            select_t= time4-time3
            temp_result = [self.__pos, self.Dom, self.best_pos, self.Ovs[::3]]
        # 让其他进程更新下一代的初始值
        temp_result = comm.bcast(temp_result if rank == 0 else None, root=0)
        self.__pos = temp_result[0].copy()
        self.Dom = temp_result[1].copy()
        self.best_pos = temp_result[2].copy()
        self.Ovs[::3] = temp_result[-1].copy()
        time2 = time.time()
        comm_t = time2-time1-select_t
        return comm_t, select_t
    
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
    
    def changeF(self, t):
        Lambda = math.exp(1-self.Gm/(1+self.Gm-t))
        self.F = self.F0 * np.exp2(Lambda)

    # normalize
    def normalize(self, data):
        pmin = np.min(data)
        pmax = np.max(data)
        _range = pmax - pmin
        # if _range==0:
        #     print(data)
        return (data - pmin)/_range
    
    # decode
    def decode(self, _x):
        _x=_x.copy()
        _x[_x < self.m_min] = 0
        _x[_x >= self.m_min] = 1
        return _x

    def obj_fun(self, ith):
        f1, f2 = 0.0, 0.0
        feats, metrics = [], []
        if ith % 3 == 0:
            cur_indv = self.__pos[ith//3].copy()
        elif ith%3==1:
            cur_indv = self.U[ith//3].copy()
        elif ith%3==2:
            cur_indv = self.V[ith//3].copy()
        decode_pos_inv = self.decode(cur_indv)
        feats = np.where(decode_pos_inv == 1)[0]  # tuple类型，取出其中的数组值
        count = len(feats)
        # f1=个体中为1的元素个数/dim, 特征占比
        f1 = - count/self.dim
        if count > 0:
            # 从原始数据中取出相应的列进行训练
            train_features = self.trainData[:, feats]
            train_labels = self.trainData[:, -1]
            test_features = self.testData[:, feats]
            test_labels = self.testData[:, -1]
            f2, pa_svm, svm_time = svm_train_2(train_features, train_labels, test_features, test_labels)
            metrics = np.append(pa_svm, [f1, f2])
            self.svm_time += svm_time
            self.Ovs[ith] = [f1, f2]
            self.updateDominateset(ith, metrics, cur_indv)

    def execution(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        h1 = ['Iteration', 'Tpr', 'Fpr', 'Precission', 'Auc', 'Ratio', 'Accuracy']
        h2 = ['svm_t', 'pj_t', 'de_t', 'comm_t'] #　各进程通信的时间
        timestamp = time.strftime('%Y-%m-%d %H%M%S', time.localtime())
        g = None
        c = None
        title = ''
        time_complexity=[]
        weight = float(sys.argv[6])  # 如果选择的是没有网络结构的DE算法，则此参数表示CR，否则表示weight
        dataset = sys.argv[11]
        start, end = self.start, self.end
        flag1 = sys.argv[3]  # network
        flag2 = sys.argv[4]  # community
        flagt, flagg=self.analysis_parameters(flag1, flag2)
        if flagt == 'yes' and flagg == 'yes':  # 有网络结构有社区结构
            # main process load graph and community, then broadcast to others
            if rank == 0:
                title = dataset + ' NetG-MODE weight_' + str(weight) + ' iterNum_' + str(self.Gm) + ' ' + timestamp
                g = Graph.Graph(flag1, weight)  # 加载网络结构
                c = community.CommunityGroup(flag2)  # 加载社区网络分组
                print('runing NetG-MODE, weight=%.2f!' % (weight))
            g = comm.bcast(g if rank == 0 else None, root=0)
            c = comm.bcast(c if rank == 0 else None, root=0)
        elif flagt == 'no' and flagg == 'no' and rank == 0:  # 无社区结构无网络结构
            title = dataset + ' MODE CR_' + str(self.CR) + ' iterNum_' + str(self.Gm) + ' ' + timestamp
            print('runing MODE, CR=%.2f!' % (self.CR))
        elif flagt == 'yes' and flagg == 'no':  # 没有社区结构，只有网络结构
            if rank == 0:
                title = dataset + ' Net-MODE weight_' + str(weight) + ' iterNum_' + str(self.Gm) + ' ' + timestamp
                g = Graph.Graph(flag1, weight)  # 加载网络结构
                print('runing Net-MODE, weight=%.2f!' % (weight))
            g = comm.bcast(g if rank == 0 else None, root=0)
        elif flagt == 'no' and flagg == 'yes':  # 没有网络结构，只有社区结构
            if rank == 0:
                title = dataset + ' Com-MODE weight_' + str(weight) + ' iterNum_' + str(self.Gm) + ' ' + timestamp
                g = Graph.Graph(flag1, weight)  # 加载网络结构
                c = community.CommunityGroup(flag2)  # 加载社区网络分组
                print('runing Com-MODE, weight=%.2f!' % (weight))
            g = comm.bcast(g if rank == 0 else None, root=0)
            c = comm.bcast(c if rank == 0 else None, root=0)
        if rank==0:
            path1 = './result_12/excels/excels ' + title + '.csv'
            path2 = './result_12/Dom/Dom '+title+'.csv'
            path3 = './result_12/timeEstimation/time ' + title + '.csv'
        for t in range(1, self.Gm+1):
            time1 = time.time()
            # 每个进程分别计算各自部分的Dom
            for i in range(start, end):
                self.mutation(i)  # 变异
                if flagt == 'no' and flagg == 'no':  # 如果选择的是无社区结构的DE算法，则选择CR作为参考
                    self.crossover_canonical(i)
                elif flagt == 'yes' and flagg=='yes':  # 如果选择的是有社区结构的，则选择pj作为参考
                    self.crossover_NetG(i, g, c)
                elif flagt == 'no' and flagg=='yes':
                    self.crossover_Community(i, g, c)
                elif flagt == 'yes' and flagg == 'no':
                    self.crossover_Network(i, g)
            # write_rows_toCsv(path, self.U[start:end],None)
            time2 = time.time()
            comm_t, select_t = self.gather_result(comm, rank, t)
            if rank==0:
                time_complexity.append([self.svm_time, self.pj_time, time2-time1+select_t, comm_t])
                self.write_Dom(t, path1, h1)  # 主进程把每次迭代的最优解写入文件
                print('finished ', t, ' iteration')
            self.changeF(t)
            self.svm_time, self.pj_time = 0.0, 0.0
        # 迭代结束时，主进程把最终的实验结果写入文件
        if rank == 0:
            self.write_Dom(t, path2, h1)
            write_rows_toCsv(path3, time_complexity, h2)

    def write_Dom(self, t, path, h):
        tt = [t] * len(self.Dom)
        temp_dom = np.c_[tt, self.Dom]
        write_rows_toCsv(path, temp_dom, h)
        return temp_dom

    # 变异                
    def mutation(self, i):
        j = k = 0
        while j == i:
            j = random.randint(0, self.size - 1)
        while k == i or k == j:
            k = random.randint(0, self.size - 1)
        u = self.get_pos(i) + self.F * (self.get_pos(j) - self.get_pos(k))
        self.U[i] = self.normalize(u)
        self.obj_fun(3*i+1)

# 无网络结构、无社区结构的交叉, 即标准DE的 crossover操作
    def crossover_canonical(self, i):
        ll = list(range(self.dim))
        random.shuffle(ll)
        u = self.U[i]
        pos = self.__pos[i]
        self.V[i] = np.array([u[j] if random.random() < self.CR else pos[j] for j in ll])
        self.obj_fun(3*i+2)
    

# 有网络结构，无社区结构的交叉
    def crossover_Network(self, i, g):
        v = np.zeros((self.dim,), float)
        Alpha = float(sys.argv[7])
        Beta = float(sys.argv[8])
        pj = 0.0
        ll = list(range(self.dim))
        random.shuffle(ll)
        u = self.U[i]
        decode_u = self.decode(u)
        pos = self.__pos[i]
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
            # 该特征节点的邻接点中被置为1的加权个数，在变异后的个体中选择元素，并在graph中找到该特征节点与对应特征节点的weight，并求和
            wjn = g.cal_weight_number(str(j), decode_u)
            pj = Alpha * math.exp(-3/djw) + Beta * math.exp(-wjn)
            time2 = time.time()
            self.pj_time += (time2-time1)
            if rand_value < self.CR*(1-pj)+pj:
                maxpv = max(u[j], pos[j])
                v[j] = maxpv  # [0.5, 1), 交叉个体选择the current feature
            elif rand_value>=self.CR*(1-pj)+pj:
                minpv = min(u[j], pos[j])
                v[j] = minpv  # [0, 0.5), 交叉个体不选择该特征
        self.V[i] = v
        self.obj_fun(3*i+2)

# 无网络结构 有社区结构的交叉
    def crossover_Community(self, i, g, c):
        v = np.zeros((self.dim,), float)
        Alpha = float(sys.argv[7])
        Gamma = float(sys.argv[9])
        pj = 0.0
        ll = list(range(self.dim))
        random.shuffle(ll)
        u = self.U[i]
        decode_u = self.decode(u)
        pos = self.__pos[i]
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
            time2 = time.time()
            self.pj_time += (time2-time1)
            if rand_value < self.CR*(1-pj)+pj:
                maxpv = max(u[j], pos[j])
                v[j] = maxpv  # [0.5, 1), 交叉个体选择the current feature
            elif rand_value>=self.CR*(1-pj)+pj:
                minpv = min(u[j], pos[j])
                v[j] = minpv  # [0, 0.5), 交叉个体不选择该特征
        self.V[i] = v
        self.obj_fun(3*i+2)

# 有网络结构 有社区结构的交叉
    def crossover_NetG(self, i, g, c):
        v = np.zeros((self.dim,), float)
        Alpha = float(sys.argv[7])
        Beta = float(sys.argv[8])
        Gamma = float(sys.argv[9])
        pj = 0.0
        ll = list(range(self.dim))
        random.shuffle(ll)
        u = self.U[i]  # 赋值地址
        decode_u = self.decode(u)
        pos = self.__pos[i]
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
            if rand_value < self.CR*(1-pj)+pj:
                maxpv = max(u[j], pos[j])
                v[j] = maxpv  # [0.5, 1), 交叉个体选择the current feature
            elif rand_value>=self.CR*(1-pj)+pj:
                minpv = min(u[j], pos[j])
                v[j] = minpv  # [0, 0.5), 交叉个体不选择该特征
        self.V[i] = v
        self.obj_fun(3*i+2)

    def select(self):
        self.reInitialize()
        self.FMaxMin()
        preindex = np.array([], dtype=int)
        selectIndex = np.array([], dtype=int)
        backup = np.array([], dtype=int)      
        # 将值分配到bins个块里面
        for m in range(self.M):
            unit = (self.Fmax[m] - self.Fmin[m]) / self.bins
            if unit == 0:
                continue
            for i in range(self.size):
                k = (self.Ovs[3*i][m] - self.Fmin[m]) / unit
                k = int(k)
                if self.Dm[m][k] == -1:
                    self.Dm[m][k] = 3*i
                elif self.Ovs[3*i][m] > self.Ovs[self.Dm[m][k]][m]:
                    self.Dm[m][k] = 3*i
                
                k = (self.Ovs[3*i+1][m] - self.Fmin[m]) / unit
                k = int(k)
                if self.Dm[m][k] == -1:
                    self.Dm[m][k] = 3*i+1
                elif self.Ovs[3*i+1][m] > self.Ovs[self.Dm[m][k]][m]:
                    self.Dm[m][k] = 3*i+1

                k = (self.Ovs[3*i+2][m] - self.Fmin[m]) / unit
                k = int(k)
                if self.Dm[m][k] == -1:
                    self.Dm[m][k] = 3*i+2
                elif self.Ovs[3*i+2][m] > self.Ovs[self.Dm[m][k]][m]:
                    self.Dm[m][k] = 3*i+2

        # 取出Dm中!=-1的值
        for m in range(self.M):
            preindex = np.append(preindex, self.Dm[m][self.Dm[m] > -1])
        preindex = list(set(preindex))  # 去重
        if len(preindex) >= self.size:
            selectIndex = np.random.choice(preindex, self.size, replace=False)
        else:
            ll = set(np.arange(0, 3*self.size))
            backup = list(ll.symmetric_difference(set(preindex)))
            surplus = self.size - len(preindex)
            surplusIndex = np.random.choice(backup, surplus, replace=False)
            selectIndex = np.append(preindex, surplusIndex)
        for i in range(self.size):
            r = selectIndex[i]
            if r % 3 == 0: # 更新个体
                self.__pos[i] = self.__pos[r//3].copy()
            elif r % 3 == 1:
                self.__pos[i] = self.U[r//3].copy()
            else:
                self.__pos[i] = self.V[r//3].copy()
            self.Ovs[3*i] = self.Ovs[r].copy() # 更新目标函数值
        

    # 更新最优解集
    def updateDominateset(self, ith, metrics, cur_indv):
        flagg = False  # 判断是否可以支配Dom中的解
        flagt = False  # 判断是否为非支配解
        if len(self.Dom) == 0:
            self.Dom = np.array([metrics])
            self.best_pos = np.array([cur_indv])
            return 0 
        delIndex = []
        for d in range(len(self.Dom)):
            flag1 = False  # 判断是否有比它大的
            flag2 = False  # 判断是否有比它小的
            for m in range(self.M):
                if self.Ovs[ith][m] > self.Dom[d][m-2]:
                    flag1 = True
                elif self.Ovs[ith][m] < self.Dom[d][m-2]:
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
            self.Dom = np.concatenate((self.Dom, [metrics]), 0)
            self.best_pos = np.concatenate((self.best_pos, [cur_indv]), 0)

    # 合并各进程的最优解集
    def MergeDominateset(self, new_Dom, new_pos):
        flagg = False  # 判断是否可以支配Dom中的解
        flagt = False  # 判断是否为非支配解
        addIndex = []
        for i in range(len(new_Dom)):
            metrics = new_Dom[i]
            delIndex = []
            for d in range(len(self.Dom)):
                flag1 = False  # 判断是否有比它大的
                flag2 = False  # 判断是否有比它小的
                for m in range(self.M):
                    if metrics[m-2] > self.Dom[d][m-2]:
                        flag1 = True
                    elif metrics[m-2] < self.Dom[d][m-2]:
                        flag2 = True
                if flag1 is True and flag2 is True:
                    flagt = True  # 表示该indv是nondominate solution
                elif flag1 is True and flag2 is False:
                    flagg = True  # 表示该indiv可支配原有解集中的解，那么应删除原解集中的对应解
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
                addIndex.append(i)
        # 如果新来的个体打败了self.Dom中的所有个体
        if len(addIndex)>0:
            self.Dom = np.concatenate((self.Dom, new_Dom[addIndex]), 0)
            self.best_pos = np.concatenate((self.best_pos, new_pos[addIndex]), 0)

    # 找到self.Ovs中的每个目标的最大最小函数值
    def FMaxMin(self):
        for m in range(self.M):
            # find the max and min in every objective
            self.Fmin[m] = np.min(self.Ovs[:, m])
            self.Fmax[m] = np.max(self.Ovs[:, m])
    
    # reset
    def reInitialize(self):
        self.Dm = -np.ones((self.M, self.bins+1), dtype=int)


def read_raw_data(path):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    time1 = time.time()
    rawData = pd.read_csv(path, header=None)
    data = rawData.values
    time2 = time.time()
    if rank == 0:
        print("read data from %s, takes up %s seconds" % (path, time2 - time1))
    return data


def write_to_csv(path, data, m):
    File = open(path, mode=m, newline='')
    writer = csv.writer(File)
    writer.writerow(data)
    File.close()


def write_rows_toCsv(path, data, h):
    df = pd.DataFrame(data)
    df.to_csv(path, mode='a', header=h)
