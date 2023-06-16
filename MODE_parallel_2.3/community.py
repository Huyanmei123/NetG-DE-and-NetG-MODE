'''
Author: lumin
Date: 2020-12-27 14:24:14
LastEditTime: 2020-12-27 14:25:02
LastEditors: Please set LastEditors
Description: load community structure
FilePath: /DE_parallel_together/community.py
'''
# %%
import numpy as np
import time

class CommunityGroup(object):
    def __init__(self, path):
        # self.community_id = id
        self.community_dict = dict()
        self.f_c_mapdict = dict()
        self.loadCommunityGroup(path)

    # 从数据文件中读取社区信息
    # 加载社区分组，社区id和图顶点id的对应dictionary
    def loadCommunityGroup(self, path):
        time_1 = time.time()
        id = 1
        if path == 'None':
            print("no community!")
            return 0
        fp = open(path, "r")
        while 1:
            line = fp.readline()
            if not line:
                break
            self.community_dict[id] = line.strip().split(" ")
            for i in range(len(self.community_dict[id])):
                self.f_c_mapdict[self.community_dict[id][i]] = id
            id += 1
        time_2 = time.time()
        print('load community group correctly!')
        print("load commuity groups take up %s seconds" % (time_2 - time_1))

    # 该特征节点所属特征组中被置为1的个数
    def cal_selected_number(self, key, _X):
        selectedNumber = 0
        cList = []
        if key in self.f_c_mapdict.keys():  # 如果该节点在社区分组中
            cid = self.f_c_mapdict[key]
            # print("该节点在第%s个社区中" % (cid))
            cList = self.community_dict[cid]  # 则获取该社区的所有节点
            # print("该社区的所有节点为：", cList)
            if len(cList) == 1:  # 如果该社区只有一个节点，则同组中被选的节点个数为0
                selectedNumber = 0
            else:
                for fi in cList:
                    if _X[int(fi)] == 1:
                        selectedNumber += 1
        return selectedNumber

    def print_c(self):
        print(self.community_dict)


if __name__ == "__main__":
    path = 'D:/MyFile/AProgram/workspace/the-first-topic-new/data/processedData/gisette-com0.75-4.txt'
    c = CommunityGroup(path)
    # c.print_c()
    x = np.array([np.random.randint(0, 2) for i in range(4971)])
    vv = c.cal_selected_number('25', x)
    print(vv)
