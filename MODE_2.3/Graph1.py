'''
Author: lumin
Date: 2020-12-27 14:23:04
LastEditTime: 2023-02-12 23:55:38
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Description: load graph structure
FilePath: /DE_parallel_together/graph.py
'''
# %%
import time


# 顶点类
class Vertex(object):
    def __init__(self, key):
        self.id = key
        self.adjList = {}

    def getId(self):
        return self.id

    # addneighbor
    def addNeighbor(self, nbr, weight):
        self.adjList[nbr] = weight

    # getadjList
    def getAdjList(self):
        return self.adjList.keys()

    # getweight
    def getWeight(self, nbr):
        return self.adjList[nbr]

    def __str__(self):
        return str(self.id) + ' connected to ' + str([x for x in self.adjList])


# graph
class Graph(object):
    def __init__(self, path, weight0):
        self.verList = {}
        self.numVertex = 0
        self.weight_degrees = {}
        self.loadFeatureGraph(path, weight0)

    def getNumVertex(self):
        return self.numVertex

    # addvertex
    def addVertex(self, key):
        newVertex = Vertex(key)
        self.verList[key] = newVertex
        self.numVertex += 1
        return True

    # getvertex
    def getVertex(self, key):
        if key in self.verList.keys():  # 判断key是否合法
            return self.verList[key]
        else:
            return None

    def getVertices(self):
        # 返回所有顶点编号
        return self.verList.keys()

    # addEdge
    def addEdge(self, key1, key2, weight):
        if key1 not in self.verList.keys():
            self.addVertex(key1)
        if key2 not in self.verList.keys():
            self.addVertex(key2)
        self.verList[key1].addNeighbor(key2, weight)
        return True

    def __iter__(self):
        return iter(self.verList.values())

    def loadFeatureGraph(self, path, weight0):
        time_1 = time.time()
        if path == 'None':
            print("no Graph!")
            return 0
        fp = open(path, "r")
        f1 = f2 = 0
        weight = 0.0
        while 1:
            line = fp.readline()
            if not line:
                break
            f1, f2, weight = line.strip().split(" ")
            weight = float(weight)
            f1, f2 = int(f1), int(f2)
            if weight >= weight0:  # 所有特征节点之间的边都是>=0.7的
                self.addEdge(f1, f2, weight)
                self.addEdge(f2, f1, weight)
        time_2 = time.time()
        print("load feature graph takes up %s seconds" % (time_2 - time_1))
        # 计算特征节点的加权度
        flag = self.cal_weight_degree()
        print('weight degree calculate ', flag)

    # 计算特征节点的加权度
    def cal_weight_degree(self):
        maxwd = 0.0  # 最大加权度
        for key in self.getVertices():
            weight_degree = 0.0
            for v in self.verList[key].getAdjList():
                weight_degree += self.verList[key].getWeight(v)
            self.weight_degrees[key]=weight_degree
            if weight_degree>maxwd:
                maxwd = weight_degree
        self.maxwd = maxwd  # 该网络中的最大加权度
        return True

    def get_weight_degree(self, key):
        if key not in self.verList.keys():
            return 0.001   # !如果该顶点没有相邻节点，则weight_degree默认置为0.001
        else:
            return self.weight_degrees[key]

    # 计算该特征节点的邻接点中被置为1的个数
    def cal_weight_number(self, key, _X):
        # time_1 = time.time()
        weight_number = 0
        if key not in self.verList.keys():
            return 0  # 如果该顶点没有相邻节点，则weight_number默认为0
        else:
            feats = self.verList[key].getAdjList()  # 找到相邻节点，并查看该个体对应位是否为1，若为1，则找到该顶点与对应节点的weight，并求和
            # print(feats)
            for v in feats:
                if _X[v] ==1:
                    # print(v)  # 该个体的对应位中第多少个元素是为1的
                    weight_number += self.verList[key].getWeight(v)
                    weight_number += (self.verList[key].getWeight(v)*self.weight_degrees[v]/self.maxwd)
        # time_2 = time.time()
        # print("cal_weight_number takes up %s seconds" % (time_2 - time_1))
        return weight_number

    # 输出图的所有边
    def print_g(self):
        for key in self.verList:
            key1 = self.getVertex(key)
            for v in key1.getAdjList():
                print('(%s, %s) weight=%s' % (key1.getId(), v, key1.getWeight(v)))


def testGraph():
    path = '../../../data/processedData/credit-card-feat_network_norm_pearson06.txt'
    g = Graph(path, 0.8)
    weight_degree = g.get_weight_degree(22)
    print('weight_degree=', weight_degree)
    # g.addEdge(1, 4, 9)
    # g.addEdge(2, 4, 6)
    # g.addEdge(1, 2, 5)
    # g.addEdge(2, 3, 1)
    # g.addEdge(1, 3, 7)
    # g.addEdge(4, 1, 9)
    # g.addEdge(4, 2, 6)
    # g.addEdge(2, 1, 5)
    # g.addEdge(3, 2, 1)
    # g.addEdge(3, 1, 7)
    # print(g.cal_weight_degree(2))
    # g.print_g()
    # ll = g.cal_weight_number(2)
    # print(ll)


if __name__ == "__main__":
    path = '../../../data/network/chen-2002-norm-pearson06.txt'
    g = Graph(path, 0.6)
    # vs = list(g.getVertices())
    # vs = list(map(int, g.getVertices()))
    vs = g.getVertices()
    # for v in vs:
    #     print(type(v))
    # print(type(list(vs)))
    # print(len(vs))
    print(vs)
    weight_degree = g.get_weight_degree(1)
    print('weight_degree=', weight_degree)
    # time1 = time.time()
    # weight_degree = g.get_weight_degree(22)
    # print('weight_degree=', weight_degree)
    # time2 = time.time()
    # print('time of getting weight_degree value: ', time2-time1)
    
    # weight_degree = g.get_weight_degree(11)
    # print('weight_degree=', weight_degree)
    # time2 = time.time()
    # print('time of getting weight_degree value: ', time2-time1)

    # weight_degree = g.get_weight_degree(2)
    # print('weight_degree=', weight_degree)
    # time2 = time.time()
    # print('time of getting weight_degree value: ', time2-time1)


# %%
# import time
# def loadFeatureGraph(path, weight0):
#     time_1 = time.time()
#     if path == 'None':
#         print("no Graph!")
#         return 0
#     fp = open(path, "r")
#     f1 = f2 = 0
#     weight = 0.0
#     g = Graph()
#     # fg_dict = dict()
#     while 1:
#         line = fp.readline()
#         if not line:
#             break
#         f1, f2, weight = line.strip().split(" ")
#         weight = float(weight)
#         if weight >= weight0:  # 所有特征节点之间的边都是>=0.7的
#             g.addEdge(f1, f2, weight)
#             g.addEdge(f2, f1, weight)
#     time_2 = time.time()
#     print("load feature graph takes up %s seconds" % (time_2 - time_1))
#     # 计算特征节点的加权度
#     weight_degree_list = []
#     for key in g.getVertices():
#         weight_degree_list.append(g.cal_weight_degree(key))
#     return g


# if __name__ == "__main__":
#     path = 'E:/MyFile/AProgram/python/workspace/the-first-topic/data/rawData/gisette/gisette-feat_network_norm_pearson06.txt'
#     # path = 'None'
#     loadFeatureGraph(path, 0.75)
