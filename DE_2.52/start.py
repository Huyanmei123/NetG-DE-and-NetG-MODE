import os

with open('./command/DEFS2.52_2.txt') as f:
# with open('./command/DEFS2.52_2_pas.txt') as f:
    lines = f.readlines()
    for command in lines:
        os.system(command)

f.close()

