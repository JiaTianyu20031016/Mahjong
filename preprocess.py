import os, json
import re

match_cnt = 0

def extract_json_from_string(text):
    dicts, stack = [], []
    for i in range(len(text)):
        if text[i] == '{':
            stack.append(i)
        elif text[i] == '}':
            begin = stack.pop()
            if not stack:
                dicts.append(json.loads(text[begin:i+1]))
    return dicts
    
    
def reformat(raw_data_dir, output_dir):
    global match_cnt
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(raw_data_dir):
        path = os.path.join(raw_data_dir, file)
        if os.path.isdir(path):
            reformat(path, output_dir)
            continue
        if not file.endswith("matches"):
            continue
        
        print(f"Processing {path}...")
        
        with open(os.path.join(raw_data_dir, file), 'r') as raw:
            lines = raw.read()
            jsons = extract_json_from_string(lines)
            for json_data in jsons:
                match_cnt += 1
                curTile = ''
                with open(os.path.join(output_dir, f"{match_cnt}.txt"), 'w') as f:
                    f.write(f"Match {match_cnt}\n")
                    for log in json_data["log"]:
                        if "output" not in log:
                            continue
                        request = log["output"]
                        type = request["display"]["action"]
                        
                        try:
                            content = {id: request["content"][str(id)].split() for id in range(4)}
                        except:
                            content = {id: request["content"][str(id)] for id in range(4)}
                        
                        if type == 'INIT':
                            f.write(f"Wind {content[0][2]}\n")
                            
                        elif type == "DEAL":
                            for id in content:
                                f.write(f"Player {id} Deal {' '.join(content[id][5:])}\n")
                        
                        elif type == "DRAW":
                            player = request["display"]["player"]
                            f.write(f"Player {player} Draw {content[player][1]}\n")
                        
                        elif type == "PLAY":
                            f.write(f"Player {content[0][1]} Play {content[0][3]}\n")
                            curTile = content[0][3]
                        
                        elif type == "CHI":
                            f.write(f"Player {content[0][1]} Chi {content[0][3]}\n")
                            f.write(f"Player {content[0][1]} Play {content[0][-1]}\n")
                            curTile = content[0][-1]
                            
                        elif type == "PENG":
                            f.write(f"Player {content[0][1]} Peng {curTile}\n")
                            f.write(f"Player {content[0][1]} Play {content[0][-1]}\n")
                            curTile = content[0][-1]
                        
                        elif type == "BUGANG":
                            f.write(f"Player {content[0][1]} BuGang {content[0][3]}\n")
                        
                        elif type == "GANG":
                            tile = request["display"]["tile"]
                            if curTile == tile:
                                f.write(f"Player {content[0][1]} Gang {curTile}\n")
                            else:
                                f.write(f"Player {content[0][1]} AnGang {tile}\n")
                            
                        elif type == "HU":
                            player = request["display"]["player"]
                            f.write(f"Player {player} Hu\n")
                            f.write(f"Score\n")
                            
                        else:
                            f.write(f"{type}\n")
                            
                        
def filter(output_dir):
    for file in os.listdir(output_dir):
        with open(os.path.join(output_dir, file), 'r') as f:
            lines = f.readlines()
            if lines[-1] != "Score\n":
                os.remove(os.path.join(output_dir, file))


def augment(data_path):
    for file in os.listdir(data_path):
        tiles = 'WTB'
        tmps = ['__W__', '__T__', '__B__']
        perm = 'BTW'
        m = {}
        for x,y in zip(tmps, perm):
            m[x] = y
        with open(os.path.join(data_path, file), encoding='UTF-8') as f:
            data = f.read()
        data = data.replace('Wind', '__X__')
        data = data.replace('BuGang', '__Y__')
        for x,y in zip(tiles, tmps):
            data = data.replace(x, y)
        for x, y in zip(tmps, perm):
            data = data.replace(x, y)
        data = data.replace('__X__', 'Wind')
        data = data.replace('__Y__', 'BuGang')
        with open(os.path.join(data_path, f"{file.split('.')[0]}-aug.txt"), 'w', encoding='UTF-8') as f:
            f.write(data)


if __name__ == '__main__':
    raw_data_path = '/data/jiaty/mahjong/raw_data/'
    data_path = '/data/jiaty/mahjong/data/'
    os.makedirs(data_path, exist_ok=True)
    reformat(raw_data_path, data_path)
    filter(data_path)   
    augment(data_path=data_path)    