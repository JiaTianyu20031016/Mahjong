import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import pickle as pkl
import math
import random
from bisect import bisect_right
from .model import ModelManager
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict


class MahjongGBDataset(Dataset):

    def __init__(self, metas, winner_only = True):
        self.metas = metas
        if winner_only:
            self.match_samples = [meta['winner_sample_num'] for meta in metas]
        else:
            self.match_samples = [meta['sample_num'] for meta in metas]
        self.matches = len(self.match_samples)
        self.samples = sum(self.match_samples)
        self.winner_only = winner_only

        t = 0
        print (f'{self.matches} matches, {self.samples} samples')
        for i in range(self.matches):
            a = self.match_samples[i]
            self.match_samples[i] = t
            t += a
        self.cache = defaultdict(dict)

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        match_id = bisect_right(self.match_samples, index, 0, self.matches) - 1
        sample_id = index - self.match_samples[match_id]
        if match_id not in self.cache['obs']:
            d = np.load(self.metas[match_id]['file'])
            self.cache['obs'][match_id] = (d['win_obs'] if self.winner_only else d['obs'])
            self.cache['mask'][match_id] = (d['win_mask'] if self.winner_only else d['mask'])
            self.cache['act'][match_id] = (d['win_act'] if self.winner_only else d['act'])
            
            #x_idx = list(range(self.cache['mask'][match_id].shape[0]))
            #y_idx = list(self.cache['act'][match_id])
            #assert np.all(self.cache['mask'][match_id][x_idx, y_idx]), 'invalid action detected!'
            
        return self.cache['obs'][match_id][sample_id], self.cache['mask'][match_id][sample_id], \
            self.cache['act'][match_id][sample_id]


def prepare_dataset(datapath, train_ratio = 0.9, winner_only=False, shuffle=True):
    import json
    with open(os.path.join(datapath, 'meta.json')) as f:
        matches = json.load(f)
    if shuffle:
        random.shuffle(matches)
    
    train_matches = matches[:int(len(matches)*train_ratio)]
    val_matches = matches[int(len(matches)*train_ratio):]
            
    print (f'train: {len(train_matches)}, validation: {len(val_matches)}')
    return MahjongGBDataset(train_matches, winner_only=winner_only), MahjongGBDataset(val_matches, winner_only=winner_only)


if __name__ == '__main__':
    logdir = '/home/jiaty/Mahjong-RL-botzone/Mahjong/model/'
    model_dir = '/home/jiaty/Mahjong-RL-botzone/Mahjong/model/checkpoint1'
    data_dir = '/data/jiaty/mahjong/supervise_data/'
    os.makedirs(model_dir, exist_ok=True)

    # Load dataset
    splitRatio = 0.9
    batchSize = 1024
    trainDataset, validateDataset = prepare_dataset(datapath=data_dir, train_ratio=0.9, winner_only=False)
    loader = DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=True)
    vloader = DataLoader(dataset=validateDataset, batch_size=batchSize, shuffle=False)

    # Load model
    manager = ModelManager(model_dir=model_dir)
    model = manager.get_model()
    version = 0
    model = model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=0.9)        # lr_scheduler

    # Train and validate
    accs = []
    validation_accs = []
    
    for e in range(10):
        print('Epoch', e)
        manager.save(model, version)
        version += 1
        correct = 0
        for i, d in enumerate(loader):
            input_dict = {'is_training': True, 'observation': d[0].cuda(), 'action_mask': d[1].cuda()}
            logits, _ = model(input_dict)
            loss = F.cross_entropy(logits, d[2].long().cuda())
            if math.isinf(loss): continue
            if i % 128 == 0:
                print('Iteration %d/%d' % (i, len(trainDataset) // batchSize + 1), 'policy_loss', loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = logits.argmax(dim=1)                                 # acc
            correct += torch.eq(pred, d[2].cuda()).sum().item()         # acc
        lr_scheduler.step()                                             # lr_scheduler
        last_lr = lr_scheduler.get_last_lr()
        acc = correct / len(trainDataset)
        accs.append(acc)
        print('train acc:', acc, ', last_lr:', last_lr)                 # log
        print('Run validation:')
        correct = 0
        for i, d in enumerate(vloader):
            input_dict = {'is_training': False, 'observation': d[0].cuda(), 'action_mask': d[1].cuda()}
            with torch.no_grad():
                logits, _ = model(input_dict)
                pred = logits.argmax(dim=1)
                correct += torch.eq(pred, d[2].cuda()).sum().item()
        acc = correct / len(validateDataset)
        print('Epoch', e + 1, 'Validate acc:', acc)
        validation_accs.append(acc)
        