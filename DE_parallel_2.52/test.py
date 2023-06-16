'''
Author: your name
Date: 2021-11-09 10:54:56
LastEditTime: 2021-11-14 23:19:23
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \DE_parallel_2.1\test.py
'''
#%%
import numpy as np
import pandas as pd
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

#%%
# # a = np.array([[]])
# b = np.random.random(6)
# a = np.array([b])
# c = np.random.random(6)
# a = np.concatenate((a,[c]), axis=0)
# d = np.concatenate(([a[0]],[b]),0)
# delIndex=[1]
# d=np.delete(d,delIndex,0)
# addIndex=[1]
# d = np.concatenate((d,a[addIndex]), 0)
# print(d)
# # %%
# type(a)
tempres = [1,2,3,4,5,6,7,8,9]
recv = comm.gather(tempres, root=0)
tempres = []
if rank==0:
    print(recv)

#%%
class A:
    def __init__(self, F):
        self.m_min = 0.5
        self.F=F

    # 解码
    def decode_fun(self, _x):
        _x = _x.copy()  # 传参传的是地址，若不复制，则会改变self.__pos的值
        # x = _x.copy()
        # x[x < self.m_min] = 0
        # x[x >= self.m_min] = 1
        # return x
        _x[_x < self.m_min] = 0
        _x[_x >= self.m_min] = 1
        return _x

    def getPa(self):
        self.X = np.random.random((3, 5))
        decx = self.decode_fun(self.X[0])
        # print(self.X[0])
        self.F = 0.99
        print(self.F)

F=0.6
aobj = A(F)
aobj.getPa()
print(F)
# X = np.random.random((3, 5))
# decx = aobj.decode_fun(X[0])
# X[0]