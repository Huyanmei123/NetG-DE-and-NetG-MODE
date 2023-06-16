#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
import random
import os
import sys

from mpi4py import MPI
from svm import svm_train_2
import Graph1
import community1
import utils

#%%
excels_dir = utils.excels_F_dir
timeEstimation_dir = utils.timeEstimation_F_dir

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

def draw_lineChart(y, ylab, dataset, alg, path):
    plt.clf()
    x = np.arange(len(y))
    plt.plot(x, y, label=alg)
    plt.xlabel('Iteration')
    plt.ylabel(ylab)
    plt.title(dataset)
    plt.legend()
    plt.savefig(path)

#%%
class DEFS:
    def __init__(self, N, dim, Gm,  F, CR, m_min):
        self.N = N
        self.dim = dim
        self.Gm = Gm
        self.F, self.F0 = F, F
        self.CR = CR
        self.m_min = m_min
        
        self.X = np.random.random((N, dim))
        self.offs = np.zeros((N, dim), dtype=float)
        self.timeCost = np.zeros((Gm+1, 3), dtype=float)  # Total time, Search time, svm time
        self.Ovs = np.zeros((2*N, 7), dtype=float)  # 一次迭代中，每个个体的7个metrics
        self.Ovs[: N] = np.array([self.obj_fun(self.X[i], 0) for i in range(N)])
        self.metrics = -np.ones((Gm+1, 7), dtype=float)  # tpr, fpr, precission, auc, ratio, accuracy, fitness
        maxIndex = np.argmax(self.Ovs[:, -1])
        self.metrics[0] = self.Ovs[maxIndex]
        
    def decode(self, x):
        dec_x = x.copy()
        dec_x[dec_x < self.m_min] = 0
        dec_x[dec_x >= self.m_min] = 1
        return dec_x
    
    # 归一化
    def normalize(self, data):
        pmin = np.min(data)
        pmax = np.max(data)
        _range = pmax - pmin
        data = (data - pmin)/_range
        return data

    # !目标函数obj_fun
    def obj_fun(self, _pos_inv, t):  # _pos_inv 个体的一组元素值，_X DE算法的所有pos值
        f1, f2 = 0.0, 0.0
        w1, w2 = -0.3, 0.7
        metrics = -np.ones(7, dtype=float)
        dec_indv = self.decode(_pos_inv)
        # obtain the index of dimensionality whose value equal to 1
        feats = np.where(dec_indv == 1)[0]
        counts = len(feats)
        f1 = counts / self.dim
        if counts!=0:  # calculate f2
            train_features, train_labels = train_data[:, feats], train_data[:, -1]
            test_features, test_labels = test_data[:, feats], test_data[:, -1]
            metrics[:4], f2, svm_t = svm_train_2(train_features, train_labels, test_features, test_labels)
            self.timeCost[t, -1] += svm_t
        metrics[-3:] = [f1, f2, w1*f1+w2*f2]
        return metrics  

    # 变异
    def mutation(self, i):
        j = k = 0
        # 保证i != j != k
        while j == k or j == i or k == i:
            j = np.random.randint(0, self.N - 1)
            k = np.random.randint(0, self.N - 1)
        v = self.X[i] + self.F * (self.X[j] - self.X[k])
        # 对变异后的个体元素进行归一化
        v = self.normalize(v)
        return v

    # 交叉 选择变异基因或者原始基因
    def crossover(self, pos, v):
        # u = np.zeros((self.dim,), float)
        ll = list(range(self.dim))
        random.shuffle(ll)
        u = np.array([v[j] if np.random.random() < self.CR else pos[j] for j in ll])
        return u

    def crossover_network(self, pos, v, Alpha, Beta, g, vers):
        # print('crossover Net-DE')
        u = np.zeros((self.dim,), float)
        decode_v = self.decode(v)
        pj = 0.0
        ll = list(range(self.dim))
        random.shuffle(ll)
        for j in ll:
            rand_value = np.random.random()
            if rand_value < self.CR:
                u[j] = v[j]
                continue
            elif v[j] == pos[j] or vers[j]==0:
                u[j] = pos[j]
                continue
            # 第j位对应的特征节点的加权度
            djw = g.get_weight_degree(str(j))
            # 该特征节点的邻接点中被置为1的加权个数，在变异后的个体中选择元素，并在graph中找该特征节点与对应特征节点的weight，并求和
            wjn = g.cal_weight_number(str(j), decode_v)
            pj = Alpha * math.exp(-3/djw) + Beta * math.exp(-wjn)
            # print('weight_degree=', djw, 'weight number=', wjn, 'com number=', njc, 'pj=', pj)
            if rand_value < self.CR*(1-pj)+pj:
                maxpv = max(v[j], pos[j])
                u[j] = maxpv  # 有更大概率选择该特征
            else:
                minpv = min(v[j], pos[j])
                u[j] = minpv  # 有更大概率不选该特征
        return u

    def crossover_netg(self, pos, v, Alpha, Beta, Gamma, g, c, vers):
        # print('crossover netg')
        u = np.zeros((self.dim,), float)
        decode_v = self.decode(v)
        pj = 0.0
        ll = list(range(self.dim))
        random.shuffle(ll)
        for j in ll:
            rand_value = np.random.random()
            if rand_value < self.CR:
                u[j] = v[j]
                continue
            elif v[j] == pos[j] or vers[j]==0:
                u[j] = pos[j]
                continue
            # 第j位对应的特征节点的加权度
            djw = g.get_weight_degree(str(j))
            # 该特征节点的邻接点中被置为1的加权个数，在变异后的个体中选择元素，并在graph中找该特征节点与对应特征节点的weight，并求和
            wjn = g.cal_weight_number(str(j), decode_v)
            if vers[j]==2:
                # 该特征节点所在的特征组中被置为1的个数， 在community中找到特征组，并获取该组的特征节点，判断u中的个体元素对应位是否为1
                njc = c.cal_selected_number(str(j), decode_v)
                pj = Alpha * math.exp(-3/djw) + Beta * math.exp(-wjn) + Gamma * math.exp(-njc)
            else:
                pj = 0.5*math.exp(-3/djw)+0.5*math.exp(-wjn)
            # print('weight_degree=', djw, 'weight number=', wjn, 'com number=', njc, 'pj=', pj)
            if rand_value < self.CR*(1-pj)+pj:
                maxpv = max(v[j], pos[j])
                u[j] = maxpv  # 有更大概率选择该特征
            else:
                minpv = min(v[j], pos[j])
                u[j] = minpv  # 有更大概率不选该特征
        return u

    # 选择  更新下一代的种群个体，全局最优解，全局最优适应值
    def selection(self, t):
        for i in range(self.N):
            ovs_u = self.obj_fun(self.offs[i], t)
            if ovs_u[-1] >= self.Ovs[i][-1]:  # 选择新个体
                self.X[i] = self.offs[i].copy()
                self.Ovs[i] = ovs_u.copy()
        maxIndex = np.argmax(self.Ovs[:, -1])
        if self.Ovs[maxIndex, -1] > self.metrics[t-1, -1]:  # 更新全局最优
            self.metrics[t] = self.Ovs[maxIndex].copy()
        else:
            self.metrics[t] = self.metrics[t-1].copy()

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
        # # Alpha, Beta, Gamma = 0.3, 0.3, 0.4
        # Alpha, Beta, Gamma = 0.5, 0.5, 0.0  
        # flag = '2'
        if flag=='1':
            algorithm = 'DE'
            print('rank %d running %s for %s' % (rank, algorithm, dataset))
        elif flag=='2':
            algorithm='Net-DE'
            g = Graph1.Graph(path_network, weight)
            ver_network = list(g.getVertices())
            vers = np.zeros(self.dim, dtype=int)
            vers[ver_network] = 1
            print('rank %d running %s for %s, weight=%.2f' % (rank, algorithm, dataset, weight))
        else:
            algorithm = 'NetG-DE'
            g = Graph1.Graph(path_network, weight)
            c = community1.CommunityGroup(path_com)
            ver_network = list(g.getVertices())  # 网络结构中的节点
            ver_com = list(c.getAllVertices())  # 社区中的节点
            vers = np.zeros(self.dim, dtype=int)
            vers[ver_network] = 1
            vers[ver_com] = 2
            print('rank %d running %s for %s, weight=%.2f' % (rank, algorithm, dataset, weight))
        for t in range(1, self.Gm+1):
            time1 = time.time()
            for i in range(self.N):
                v = self.mutation(i)
                if flag=='1':
                    self.offs[i] = self.crossover(self.X[i], v)
                elif flag=='2':
                    self.offs[i] = self.crossover_network(self.X[i], v, Alpha, Beta, g, vers)
                elif flag=='3':
                    self.offs[i] = self.crossover_netg(self.X[i], v, Alpha, Beta, Gamma, g, c, vers)
            self.selection(t)
            time2 = time.time()
            total_t = time2 - time1
            self.timeCost[t, :2] = [total_t, total_t - self.timeCost[t, -1]]
            print('iteration %d, ratio=%.4f, accuracy=%.4f, fitness=%.4f' % (t, self.metrics[t,-3], self.metrics[t,-2], self.metrics[t,-1]))
        # write results to csv files
        timestamp = time.strftime("%Y-%m-%d %H%M%S", time.localtime())
        str1 = '{} {} weight_{} Iteration_{} N_{} F_{} rank_{} {}'.format(dataset, algorithm, weight, self.Gm, self.N, self.F, rank, timestamp)
        # str1 = dataset +' ' + algorithm + ' weight_' + str(weight) + ' Iteration_' + str(self.Gm) + ' N_' + str(self.N) + ' rank_' + str(rank) + ' ' + timestamp
        fit_path = '{}excels {}.csv'.format(excels_dir, str1)
        time_path = '{}timeEstimation {}.csv'.format(timeEstimation_dir, str1)
        fit_h = ['Tpr', 'Fpr', 'Precission', 'Auc', 'Ratio', 'Accuracy', 'Fitness']
        time_h = ['Total_t', 'Search_t', 'Svm_t']
        write_rows(fit_path, self.metrics, fit_h)
        write_rows(time_path, self.timeCost, time_h)
        print('rank {} finished {} for {}, weight={}'.format(rank, algorithm, dataset, weight))

#%%
if __name__ == "__main__":
    np.random.seed(int(np.random.random()*100000))
    dim = len(train_data[0])-1
    iteration = int(sys.argv[9])
    dataset = sys.argv[10]
    N=int(sys.argv[12])
    # iteration = 100
    # dataset = 'Shipp'
    # N=200
    m_min = 0.5
    F, CR = float(sys.argv[13]), 0.6
    de = DEFS(N, dim, iteration, F, CR, m_min)
    de.execution(dataset)