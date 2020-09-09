import torch
import random
from replay import ReplayBuffer, sample_n_unique
from PIL import Image
import numpy as np
from utils.helpers import process_state
import gym

replay_memory = ReplayBuffer(1000, 4)
env = gym.envs.make('BreakoutDeterministic-v4')

state = env.reset()
state = process_state(state)

for i in range(2000):
    last_idx = replay_memory.store_frame(state)
    action = np.random.choice(np.arange(env.action_space.n), p=[.1, .3, .3, .3])
    next_state, reward, done, _ = env.step(action)
    print(done)
    replay_memory.store_effect(last_idx, action, reward, done)
    next_state = process_state(next_state)
    state = next_state
