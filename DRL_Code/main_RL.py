from D3QN.D3QN_trainer import *
from tqdm import tqdm
import sys
sys.path.append("env") # Adding the 'env' directory to the path to import surrogate model (env.net_middle).
import random
import numpy as np
import torch

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

history_values = {}

def trainer(x, y,z):
    from env.Env_8conv import env  # Importing a custom environment.

    print("x={}, y={} ,z={}".format( x,y,z))

    env = env()
    trainer = DQNTrainer(env=env, config=pytorch_config)
    config = trainer.config

    for i in tqdm(range(config['max_iteration'] + 1)):
        reward, done = trainer.train(termtype="fixed", start=x, m1=y, m2=z)

    print("DRL training finished. Please use plt_reward.py to observe the learning curves.")

if __name__ == "__main__":

    setup_seed(1)
    x = 0  # Start position x-coordinate
    y = 0  # Must-visit point 1 y-coordinate
    z = 0  # Must-visit point 2 y-coordinate

    trainer(x, y, z)


