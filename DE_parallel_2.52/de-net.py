'''
Author: lumin
Date: 2020-12-27 14:17:14
LastEditTime: 2021-11-13 22:08:25
LastEditors: Please set LastEditors
Description: 函数入口
FilePath: /DE_parallel_together/de-net.py
'''
import sys
from DifferentialEvolutionaryAlgorithm import DE


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
    x_max = 1.0
    x_min = 0.05
    m_min = 0.5
    # F = 0.6
    CR = float(sys.argv[5])  # 如果选择的是没有网络结构的DE算法，则此参数表示CR，否则表示weight
    runs = int(sys.argv[12])
    for r in range(runs):
        F = 0.6
        de = DE(size, dim, Gm, x_min, x_max, F, CR, m_min)
        de.execution()
        if rank == 0:
            print(r, " runs finished successfully!")


if __name__ == "__main__":
    execution_DE()
