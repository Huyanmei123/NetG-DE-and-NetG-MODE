import os

with open('./command/g-netg-2.3.txt') as f:
    lines = f.readlines()
    for command in lines:
        os.system(command)

f.close()
