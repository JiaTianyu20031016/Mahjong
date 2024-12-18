from multiprocessing import Process
import numpy as np
import torch

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolClient
from env import MahjongGBEnv
from feature import FeatureAgent
from model import CNNModel
import os

model_path = '/home/jiaty/Mahjong-RL-botzone/Mahjong/model/checkpoint/'

def play(model_path):

    model = CNNModel()
        
    # load initial model

    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    # collect data
    env = MahjongGBEnv(config = {'agent_clz': FeatureAgent})
    policies = {player : model for player in env.agent_names} # all four players use the latest model

    obs = env.reset()
    print(sorted(env.agents[0].hand))
    episode_data = {agent_name: {
        'state' : {
            'observation': [],
            'action_mask': []
        },
        'action' : [],
        'reward' : [],
        'value' : []
    } for agent_name in env.agent_names}

    done = False
    while not done:
        # each player take action
        actions = {}
        values = {}
        for agent_name in obs:
            agent_data = episode_data[agent_name]
            state = obs[agent_name]
            agent_data['state']['observation'].append(state['observation'])
            agent_data['state']['action_mask'].append(state['action_mask'])
            state['observation'] = torch.tensor(state['observation'], dtype = torch.float).unsqueeze(0)
            state['action_mask'] = torch.tensor(state['action_mask'], dtype = torch.float).unsqueeze(0)
            model.train(False) # Batch Norm inference mode
            with torch.no_grad():
                logits, value = model(state)
                action_dist = torch.distributions.Categorical(logits = logits)
                action = action_dist.sample().item()
                value = value.item()
            if agent_name == 'player_1' :
                    if env.agents[0].action2response(action) != 'Pass':
                            print(env.agents[0].action2response(action))
                            print(sorted(env.agents[0].hand))
                            print('-' * 50)
            
            actions[agent_name] = action
            values[agent_name] = value
            agent_data['action'].append(actions[agent_name])
            agent_data['value'].append(values[agent_name])
        # interact with env
        next_obs, rewards, done = env.step(actions)
        for agent_name in rewards:
            episode_data[agent_name]['reward'].append(rewards[agent_name])
        obs = next_obs
        
    print(episode_data['player_1']['reward'])


def eval(model_paths, total_match):
    
    models = [CNNModel()] * 4
        
    # load initial model
    for i in range(4):
        state_dict = torch.load(model_paths[i])
        models[i].load_state_dict(state_dict)

    # collect data
    env = MahjongGBEnv(config = {'agent_clz': FeatureAgent})
    policies = {player : models[i] for i, player in enumerate(env.agent_names)} # all four players use the latest model

    scores = {player : 0 for player in env.agent_names}
    for id in range(total_match):
        obs = env.reset()
        episode_data = {agent_name: {
            'state' : {
                'observation': [],
                'action_mask': []
            },
            'action' : [],
            'reward' : [],
            'value' : []
        } for agent_name in env.agent_names}

        done = False
        while not done:
            # each player take action
            actions = {}
            values = {}
            for agent_id, agent_name in enumerate(obs):
                agent_data = episode_data[agent_name]
                state = obs[agent_name]
                agent_data['state']['observation'].append(state['observation'])
                agent_data['state']['action_mask'].append(state['action_mask'])
                state['observation'] = torch.tensor(state['observation'], dtype = torch.float).unsqueeze(0)
                state['action_mask'] = torch.tensor(state['action_mask'], dtype = torch.float).unsqueeze(0)
                models[agent_id].train(False) # Batch Norm inference mode
                with torch.no_grad():
                    logits, value = models[agent_id](state)
                    action_dist = torch.distributions.Categorical(logits = logits)
                    action = action_dist.sample().item()
                    value = value.item()
                
                actions[agent_name] = action
                values[agent_name] = value
                agent_data['action'].append(actions[agent_name])
                agent_data['value'].append(values[agent_name])
            # interact with env
            next_obs, rewards, done = env.step(actions)
            for agent_name in rewards:
                episode_data[agent_name]['reward'].append(rewards[agent_name])
            obs = next_obs
            
        for agent_name in rewards:
            scores[agent_name] += rewards[agent_name]
    
    print(f"{total_match} matches completed, scores for participants are:")
    for agent_name in scores:
        print(f"{agent_name}: {scores[agent_name]}")
        
        
if __name__ == '__main__':
    model_ids = [7, 30, 60, 84]
    model_paths = [os.path.join(model_path, f"model_{model_id}.pt") for model_id in model_ids]
    eval(model_paths=model_paths, total_match=100)