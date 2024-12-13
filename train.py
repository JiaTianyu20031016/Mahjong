from replay_buffer import ReplayBuffer
from actor import Actor
from learner import Learner
from streaming.base.util import clean_stale_shared_memory

if __name__ == '__main__':
    clean_stale_shared_memory()
    
    config = {
        'replay_buffer_size': 50000,
        'replay_buffer_episode': 400,
        'model_pool_size': 20,
        'model_pool_name': 'model-pool',
        'num_actors': 24,
        'episodes_per_actor': 100000,
        'gamma': 0.98,
        'lambda': 0.95,
        'min_sample': 200,
        'batch_size': 512,
        'epochs': 5,
        'clip': 0.2,
        'lr': 1e-4,
        'value_coeff': 1,
        'entropy_coeff': 0.01,
        'device': 'cuda',
        'ckpt_save_interval': 3000,
        'ckpt_save_path': '/root/jiaty/Mahjong-RL-botzone/checkpoint/nofan_nopunish/'
    }
    
    replay_buffer = ReplayBuffer(config['replay_buffer_size'], config['replay_buffer_episode'])
    
    actors = []
    for i in range(config['num_actors']):
        config['name'] = 'Actor-%d' % i
        actor = Actor(config, replay_buffer)
        actors.append(actor)
    learner = Learner(config, replay_buffer)
    
    for actor in actors: actor.start()
    learner.start()
    
    for actor in actors: actor.join()
    learner.terminate()