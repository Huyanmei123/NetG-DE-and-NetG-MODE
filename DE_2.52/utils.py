import os

result_dir = './result/'
excels_dir = result_dir+'excels/'
Figs_dir = result_dir + 'Figs/'
timeEstimation_dir = result_dir + 'timeEstimation/'

# result_d_dir = './result_Delta/'
# excels_d_dir = result_d_dir+'excels/'
# timeEstimation_d_dir = result_d_dir+'timeEstimation/'

# result_N_dir = './result_N/'
# excels_N_dir = result_N_dir+'excels/'
# timeEstimation_N_dir = result_N_dir+'timeEstimation/'

# result_F_dir = './result_F/'
# excels_F_dir = result_F_dir+'excels/'
# timeEstimation_F_dir = result_F_dir+'timeEstimation/'

# result_CR_dir = './result_CR/'
# excels_CR_dir = result_CR_dir+'excels/'
# timeEstimation_CR_dir = result_CR_dir+'timeEstimation/'

# dirs = [result_dir, excels_dir, Figs_dir, timeEstimation_dir, result_N_dir, excels_N_dir, timeEstimation_N_dir, result_d_dir, excels_d_dir, timeEstimation_d_dir, result_F_dir, excels_F_dir, timeEstimation_F_dir, result_CR_dir, excels_CR_dir, timeEstimation_CR_dir]

dirs = [result_dir, excels_dir, Figs_dir, timeEstimation_dir]

for dir in dirs:
    if not os.path.isdir(dir):
        os.makedirs(dir)
        print('create ', dir, ' successfully!')
