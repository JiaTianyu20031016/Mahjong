from multiprocessing import Process
import numpy as np
import torch

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolClient
from env import MahjongGBEnv
from feature import FeatureAgent
from model import CNNModel



model = CNNModel()
    
# load initial model

state_dict = torch.load('/root/jiaty/Mahjong-RL-botzone/framework/checkpoint/model_60528.pt')
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
