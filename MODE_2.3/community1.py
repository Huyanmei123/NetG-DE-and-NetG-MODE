'''
Author: lumin
Date: 2020-12-27 14:24:14
LastEditTime: 2020-12-27 14:25:02
LastEditors: Please set LastEditors
Description: load community structure
FilePath: \DE_parallel_together\community.py
'''
# %%
import time
import numpy as np


class CommunityGroup(object):
    def __init__(self, path):
        # self.community_id = id
        self.community_dict = dict()
        self.f_c_mapdict = dict()
        self.ver_com = set()
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
            self.community_dict[id] = list(map(int, line.strip().split(" ")))
            for i in range(len(self.community_dict[id])):
                self.f_c_mapdict[self.community_dict[id][i]] = id
                self.ver_com.add(self.community_dict[id][i])
            id += 1
        time_2 = time.time()
        print('load community group correctly!')
        print("load commuity groups take up %s seconds" % (time_2 - time_1))

    # 该特征节点所属特征组中被置为1的个数
    def cal_selected_number(self, key, _X):
        selectedNumber = 0
        cList = []
        if key in self.f_c_mapdict.keys():  # 如果该节点在社区分组中,则计算同一个社区中被选中的个数
            cid = self.f_c_mapdict[key]
            # print("该节点在第%s个社区中" % (cid))
            cList = self.community_dict[cid]  # 则获取该社区的所有节点
            # print("该社区的所有节点为：", cList)
            for fi in cList:
                # if _X[int(fi)] == 1:
                if _X[fi] == 1:
                    # print("该个体第%s个元素为1" % (fi))
                    selectedNumber += 1
        return selectedNumber

    def print_c(self):
        print(self.community_dict)

    def getAllVertices(self):
        return self.ver_com

    def getCommunities(self):
        return self.community_dict
    
    def getCommunityId(self):
        return self.community_dict.keys()

    def getOneNode_perCom(self):
        features = []
        for key in self.community_dict.keys():
            features.append(np.random.choice(self.community_dict[key], size=1, replace=False)[0])
        return features

    def getOneNode_specified(self, coms):
        features = []
        for key in coms:
            features.append(np.random.choice(self.community_dict[key], size=1, replace=False)[0])
        return features

def getFc_pea(fc_pea_path):
    # fc_pea_path = '../../../data/f_c_pearson/alizadeh-2000-v1-norm-fc-pearson.txt'
    f = open(fc_pea_path, 'r')
    contents = [line.strip().split('\t') for line in f.readlines()]
    fc_pea_dict = {}
    for i in range(len(contents)):
        ll = contents[i][0].split(' ')
        ll[1] = float(ll[1])
        fc_pea_dict[int(ll[0])] = ll[1]
    return fc_pea_dict

if __name__ == "__main__":
    # path = '../../../data/com/chen-norm-com0.60.txt'
    path = '../../../data/com/alizadeh-norm-com0.60.txt'
    fc_p = '../../../data/f_c_pearson/alizadeh-2000-v1-norm-fc-pearson.txt'
    fc_pea_dict = getFc_pea(fc_p)
    # path = 'None'
    c = CommunityGroup(path)
    # c.print_c()
    # x = np.array([np.random.randint(0, 2) for i in range(4971)])
    # vv = c.cal_selected_number('275', x)
    # vv = c.getVertexCommunity("176")
    # print(vv)

    # ver_com = list(map(int, c.getAllVertices()))
    # ver_com.sort()
    # print(ver_com)

    # # 从每个社区中随机选择一个节点
    # c.getCommunityId()
    # fs = c.getOneNode_perCom()
    # print(len(fs))

    # 从指定社区中分别选择一个节点
    # coms = list(c.getCommunities())
    # a. 获取所有社区
    coms = c.getCommunities()
    # b. 计算每个社区跟label的相关性，并降序排列
    coms_num = len(coms)
    
    fc_coms, comId = np.zeros(coms_num, dtype=float), 0
    for key in coms:
        print(key, coms[key])
        for j in range(len(coms[key])):
            # print(coms[key][j])
            fc_coms[comId] += fc_pea_dict[j]
        comId += 1
    print(fc_coms)
    fc_coms[fc_coms<0]=0.0
    index_dec = np.argsort(-fc_coms)
    print('index_dec: ', index_dec)
    print('dec fc_coms: ', fc_coms[index_dec])
    # c. 若社区总数小于等于10，则从所有社区中各随机选择一个节点；否则从前10%的社区中各取一个节点，并用轮盘赌方法选择40%的社区，从其中各取一个节点
    top_nums = int(coms_num*0.1)
    # index_top = index_dec[:top_nums]
    # print(index_top)
    minp, maxp = min(fc_coms[top_nums:]), max(fc_coms[top_nums:])
    sump = np.sum(fc_coms[top_nums:])
    p = fc_coms[top_nums:]/sump
    index_rand = np.random.choice(index_dec[top_nums:],size=int(coms_num*0.4), replace=False, p=p)+1
    # print(index_rand)
    index_select = np.append(index_dec[:top_nums], index_rand)
    print('index_select: ', index_select)
    fs = c.getOneNode_specified(index_select)
    print(fs)

#%%
def test(): # 测试：从网络结构中读出来的社区节点和community结构中获取的是否一致
    path = '../../../data/processedData/credit-card-com0.8.txt'
    f = open(path, 'r')
    ver_com = set()
    while 1:
        line = f.readline()
        if not line:
            break
        ll = line.strip().split(' ')
        for v in ll:
            ver_com.add(v)
    # print(len(ver_com), ver_com)
    ver_com = list(map(int, ver_com))
    ver_com.sort()
    print(len(ver_com))
# test()
#%%
# import numpy as np
# def getVertices_indp():
#     ver_com = [54, 27, 39, 44, 17, 53, 11, 48, 23, 29, 43, 63, 70, 45, 49, 64, 10, 32, 38, 13, 31, 52, 72, 16, 21, 47, 50, 19, 37, 36, 55, 28, 56, 24, 46, 41, 30, 40, 65, 12]
#     v_all = np.arange(87)
#     ver_indp = np.delete(v_all, ver_com)
#     print(ver_indp)

# #%%
# x =np.array([[54, 27, 39, 44, 17, 53, 11, 48], [23, 29, 43, 63, 70, 45, 49, 64]], dtype=object)
# feas = [0, 2, 5]
# x[:, feas] = np.random.uniform(low=0.5, high=1.0, size=(2, len(feas)))
# x