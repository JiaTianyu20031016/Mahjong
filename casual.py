import os

for file in os.listdir('/root/jiaty/Mahjong-RL-botzone/framework/data/'):
    if file.endswith("perm.txt"):
        os.remove(os.path.join('/root/jiaty/Mahjong-RL-botzone/framework/data/', file))