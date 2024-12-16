import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import pickle as pkl
import math
import random
from bisect import bisect_right
from model import ModelManager
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict


class MahjongGBDataset(Dataset):

    def __init__(self, indices, samples):
        self.match_samples = samples
        self.total_matches = len(self.match_samples)
        self.total_samples = sum(self.match_samples)
        self.matches = len(self.match_samples)
        self.samples = sum(self.match_samples)
        self.indices = indices
        t = 0
        print (f'{self.matches} matches, {self.samples} samples')
        for i in range(self.matches):
            a = self.match_samples[i]
            self.match_samples[i] = t
            t += a
        self.cache = {'obs': [None] * self.matches, 'mask': [None] * self.matches, 'act': [None] * self.matches}

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        match_id = bisect_right(self.match_samples, index, 0, self.matches) - 1
        sample_id = index - self.match_samples[match_id]
        if self.cache['obs'][match_id] is None:
            d = np.load(os.path.join('/root/jiaty/Mahjong-RL-botzone/framework/supervise_data', '%d.npz' % (self.indices[match_id])))
            for k in d:
                self.cache[k][match_id] = d[k]
        return self.cache['obs'][match_id][sample_id], self.cache['mask'][match_id][sample_id], \
               self.cache['act'][match_id][sample_id]


def prepare_dataset(datapath, train_ratio = 0.9, winner_only=False, shuffle=True):
    import json
    train_indices = []
    train_samples = []
    validation_indices = []
    validation_samples = []
    with open(os.path.join(datapath, 'meta.json')) as f:
        matches = json.load(f)
    if shuffle:
        random.shuffle(matches)
    matches_train = matches[:int(len(matches)*train_ratio)]
    matches_val = matches[int(len(matches)*train_ratio):]
    data_train = defaultdict(list)
    data_val = defaultdict(list)
    
    for meta in matches_train:
        d = np.load(meta['file'])
        for k in d:
            data_train[k].append(d[k][meta['winner']] if winner_only else )
    for meta in matches_val:
        d = np.load(meta['file'])
        for k in d:
            data_val[k].append(d[k])
        
    if winner_only:
        
        
    
    for idx, match in enumerate(matches):
        if random.random() < train_ratio:
            train_indices.append(i)
            train_samples.append(sample)
        else:
            validation_indices.append(i)
            validation_samples.append(sample)
            
    print (f'train: {len(train_indices)}, validation: {len(validation_indices)}')
    return MahjongGBDataset(train_indices, train_samples), MahjongGBDataset(validation_indices, validation_samples)


if __name__ == '__main__':
    logdir = './model/'
    model_dir = '/root/jiaty/Mahjong-RL-botzone/framework/model/checkpoint/'
    os.makedirs(model_dir, exist_ok=True)

    # Load dataset
    splitRatio = 0.9
    batchSize = 1024
    trainDataset, validateDataset = train_validation_split('/root/jiaty/Mahjong-RL-botzone/framework/supervise_data', 0.9)
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
    
    for e in range(150):
        print('Epoch', e)
        manager.save(model, version)
        version += 1
        correct = 0
        for i, d in enumerate(loader):
            input_dict = {'is_training': True, 'observation': d[0].cuda(), 'action_mask': d[1].cuda()}
            logits, _ = model(input_dict)
            loss = F.cross_entropy(logits, d[2].long().cuda())
            if loss > 1e10: continue
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
        if (e + 1) % 10 == 0:
            with open("acc.pkl", 'wb') as f:
                pkl.dump({"train": accs, "validation": validation_accs}, f)
        