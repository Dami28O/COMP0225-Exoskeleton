import numpy as np
action = np.zeros(71)
action[-1] = 1
action[-2] = 1
from myosuite.utils import gym
env = gym.make('myoTSA-v0')
env.reset()
# env.get_randomized_initial_state()
for _ in range(100000):
    env.mj_render()
    env.step(action) 
    # env.get_randomized_initial_state()
env.close()