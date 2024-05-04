from D3QN.D3QN_trainer import *
from tqdm import tqdm
import sys
sys.path.append("env")
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

def random_policy(x, y,z):
    from env.Env_8conv import env

    print("x={}, y={} ,z={}".format( x,y,z))

    env = env(start=5, m1=5, m2=5)
    trainer = DQNTrainer(env=env, config=pytorch_config)
    config = trainer.config

    reward_rp = []

    for i in tqdm(range(config['max_iteration'] + 1)):
        obs = env.reset()  # Reset environment for a new episode
        for i in range(1000):
            act = random.randint(0, 1)  # Random action/policy
            obs, reward, done, info = env.step(act)
            if done == True:
                break  # Exit loop if the episode is done
        reward_rp.append(reward)
        np.savetxt("RL_opt_record\\reward_rp", reward_rp)

    print("The random policy execution has ended. Please use plt_reward.py to observe the learning curves.")

if __name__ == "__main__":

    setup_seed(1)
    x = 0
    y = 0
    z = 0

    random_policy(x, y, z,)


