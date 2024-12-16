import json

import numpy as np

from feature import FeatureAgent

from itertools import permutations
import math
import os


l = []

data_path = '/root/jiaty/Mahjong-RL-botzone/framework/data'
save_path = '/root/jiaty/Mahjong-RL-botzone/framework/supervise_data'


def filterData(obs, actions):
    newobs = [[] for i in range(4)]
    newactions = [[] for i in range(4)]
    for i in range(4):
        for j, o in enumerate(obs[i]):
            if o['action_mask'].sum() > 1:  # ignore states with single valid action (Pass)
                newobs[i].append(o)
                newactions[i].append(actions[i][j])
    return newobs, newactions


def saveData(obs, actions, winner, path):
    assert [len(x) for x in obs] == [len(x) for x in actions], 'obs actions not matching!'
    l.append(
        {
            "file": path,
            "sample_num": sum([len(x) for x in obs]),
            "winner": winner
        }
    )
    np.savez(path
             , obs=np.stack([x['observation'] for i in range(4) for x in obs[i]]).astype(np.int8)
             , mask=np.stack([x['action_mask'] for i in range(4) for x in obs[i]]).astype(np.int8)
             , act=np.array([x for i in range(4) for x in actions[i]])
             , win_obs=np.stack([x['observation'] for x in obs[winner]]).astype(np.int8)
             , win_mask=np.stack([x['action_mask'] for x in obs[winner]]).astype(np.int8)
             , win_act=np.array([x for x in actions[winner]])
             )
                
                        
def process(file, savepath):
    with open(f'{data_path}/{file}', encoding='UTF-8') as f:
        line = f.readline()
        obs = [[] for i in range(4)]
        actions = [[] for i in range(4)]
        winner = -1
        while line:
            t = line.split()
            if len(t) == 0:
                line = f.readline()
                continue
            if t[0] == 'Match':
                agents = [FeatureAgent(i) for i in range(4)]
            elif t[0] == 'Wind':
                for agent in agents:
                    agent.request2obs(line)
            elif t[0] == 'Player':
                p = int(t[1])
                if t[2] == 'Deal':
                    agents[p].request2obs(' '.join(t[2:]))
                elif t[2] == 'Draw':
                    for i in range(4):
                        if i == p:
                            obs[p].append(agents[p].request2obs(' '.join(t[2:])))
                            actions[p].append(0)
                        else:
                            agents[i].request2obs(' '.join(t[:3]))
                elif t[2] == 'Play':
                    actions[p].pop()
                    actions[p].append(agents[p].response2action(' '.join(t[2:])))
                    for i in range(4):
                        if i == p:
                            agents[p].request2obs(line)
                        else:
                            obs[i].append(agents[i].request2obs(line))
                            actions[i].append(0)
                    curTile = t[3]
                elif t[2] == 'Chi':
                    actions[p].pop()
                    actions[p].append(agents[p].response2action('Chi %s %s' % (curTile, t[3])))
                    for i in range(4):
                        if i == p:
                            obs[p].append(agents[p].request2obs('Player %d Chi %s' % (p, t[3])))
                            actions[p].append(0)
                        else:
                            agents[i].request2obs('Player %d Chi %s' % (p, t[3]))
                elif t[2] == 'Peng':
                    actions[p].pop()
                    actions[p].append(agents[p].response2action('Peng %s' % t[3]))
                    for i in range(4):
                        if i == p:
                            obs[p].append(agents[p].request2obs('Player %d Peng %s' % (p, t[3])))
                            actions[p].append(0)
                        else:
                            agents[i].request2obs('Player %d Peng %s' % (p, t[3]))
                elif t[2] == 'Gang':
                    actions[p].pop()
                    actions[p].append(agents[p].response2action('Gang %s' % t[3]))
                    for i in range(4):
                        agents[i].request2obs('Player %d Gang %s' % (p, t[3]))
                elif t[2] == 'AnGang':
                    actions[p].pop()
                    actions[p].append(agents[p].response2action('AnGang %s' % t[3]))
                    for i in range(4):
                        if i == p:
                            agents[p].request2obs('Player %d AnGang %s' % (p, t[3]))
                        else:
                            agents[i].request2obs('Player %d AnGang' % p)
                elif t[2] == 'BuGang':
                    actions[p].pop()
                    actions[p].append(agents[p].response2action('BuGang %s' % t[3]))
                    for i in range(4):
                        if i == p:
                            agents[p].request2obs('Player %d BuGang %s' % (p, t[3]))
                        else:
                            obs[i].append(agents[i].request2obs('Player %d BuGang %s' % (p, t[3])))
                            actions[i].append(0)
                elif t[2] == 'Hu':
                    actions[p].pop()
                    actions[p].append(agents[p].response2action('Hu'))
                    winner = p
                # Deal with Ignore clause
                if t[2] in ['Peng', 'Gang', 'Hu']:
                    for k in range(5, 15, 5):
                        if len(t) > k:
                            p = int(t[k + 1])
                            if t[k + 2] == 'Chi':
                                actions[p].pop()
                                actions[p].append(agents[p].response2action('Chi %s %s' % (curTile, t[k + 3])))
                            elif t[k + 2] == 'Peng':
                                actions[p].pop()
                                actions[p].append(agents[p].response2action('Peng %s' % t[k + 3]))
                            elif t[k + 2] == 'Gang':
                                actions[p].pop()
                                actions[p].append(agents[p].response2action('Gang %s' % t[k + 3]))
                            elif t[k + 2] == 'Hu':
                                actions[p].pop()
                                actions[p].append(agents[p].response2action('Hu'))
                        else:
                            break
            elif t[0] == 'Score':
                obs, actions = filterData(obs, actions)
                saveData(obs=obs, actions=actions, winner=winner, path=savepath)
            else:
                print ('drop:', line)
            line = f.readline()


if __name__ == '__main__':
    os.makedirs(save_path, exist_ok=True)
    
    for file in os.listdir(data_path):
        if not file.endswith('.txt'):
            continue
        print(file)
        process(file, os.path.join(save_path, file.replace('.txt', '.npz')))


    with open(os.path.join(save_path, 'meta.json'), 'w') as f:
        json.dump(l, f)