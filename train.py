from actor import Actor
from learner import Learner
from replay_buffer import ReplayBuffer
from streaming.base.util import clean_stale_shared_memory
import os

if __name__ == '__main__':
    clean_stale_shared_memory()
    os.makedirs('model/checkpoint', exist_ok=True)
    
    config = {
        'replay_buffer_size': 5000,
        'replay_buffer_episode': 400,
        'model_pool_size': 40,
        'model_pool_name': 'model-pool',
        'num_actors': 20,
        'episodes_per_actor': 1000,
        'gamma': 0.98,
        'lambda': 0.95,
        'min_sample': 512,
        'batch_size': 512,
        'epochs': 2,
        'clip': 0.2,
        'lr': 1e-3,
        'value_coeff': 1,
        'entropy_coeff': 0.01,
        'device': 'cuda',
        'learner-device': 'cuda',
        'actor-device': 'cuda',
        'ckpt_save_interval': 50,
        'ckpt_save_path': 'model/checkpoint'
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
