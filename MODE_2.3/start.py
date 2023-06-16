import os

with open('./command/MODE2_pas.txt') as f:
    lines = f.readlines()
    for command in lines:
        os.system(command)

f.close()
