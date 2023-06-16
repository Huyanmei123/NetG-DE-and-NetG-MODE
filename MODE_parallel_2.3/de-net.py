'''
Author: lumin
Date: 2020-12-27 14:17:14
LastEditTime: 2021-11-13 22:40:13
LastEditors: Please set LastEditors
Description: 函数入口
FilePath: \DE_parallel_together\de-net.py
'''
import sys
from MODE import DE


from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()


def execution_DE():
    print('rank %d: ' % rank)
    dataset = sys.argv[11]
    if dataset == 'Credit':
        dim = 87
    elif dataset == 'Arcene':
        dim = 9961
    elif dataset == 'Dexter':
        dim = 11035
    elif dataset == 'Gisette':
        dim = 4971
    size = 200
    Gm = int(sys.argv[10])
    m_min = 0.5
    F = 0.6
    M = 2
    parameter = float(sys.argv[5])  # 如果选择的是没有网络结构的DE算法，则此参数表示CR，否则表示weight
    runs = int(sys.argv[12])
    for r in range(runs):
        de = DE(size, dim, Gm, M, F, parameter, m_min)
        de.execution()
        if rank==0:
            print(r, ' runs finished successfully!')

if __name__ == "__main__":
    execution_DE()
