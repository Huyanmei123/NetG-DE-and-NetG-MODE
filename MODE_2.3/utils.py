import os

result_dir = './result/'
Dom_dir = result_dir + 'Dom/'
timeEstimation_dir = result_dir + 'timeEstimation/'

# result_N_dir = './result_N/'
# Dom_N_dir = result_N_dir + 'Dom/'
# timeEstimation_N_dir = result_N_dir + 'timeEstimation/'

# result_F_dir = './result_F/'
# Dom_F_dir = result_F_dir + 'Dom/'
# timeEstimation_F_dir = result_F_dir + 'timeEstimation/'

# result_d_dir = './result_Delta/'
# Dom_d_dir = result_d_dir + 'Dom/'
# timeEstimation_d_dir = result_d_dir + 'timeEstimation/'

# result_cr_dir = './result_CR/'
# Dom_cr_dir = result_cr_dir + 'Dom/'
# timeEstimation_cr_dir = result_cr_dir + 'timeEstimation/'


# dirs = [result_dir, Dom_dir, timeEstimation_dir, result_N_dir, Dom_N_dir, timeEstimation_N_dir, result_d_dir, Dom_d_dir, timeEstimation_d_dir, result_F_dir, Dom_F_dir, timeEstimation_F_dir, result_cr_dir, Dom_cr_dir, timeEstimation_cr_dir]

dirs = [result_dir, Dom_dir, timeEstimation_dir]

for dir in dirs:
    if not os.path.isdir(dir):
        os.makedirs(dir)
        print('create ', dir, ' successfully!')

#%%

# for i in range(-4, 0):
#     print(i)
# %%
