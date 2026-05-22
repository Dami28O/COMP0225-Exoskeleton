# import gymnasium as gym
from myosuite.utils import gym

# Load the simplest 2D model (22 muscles, no exoskeleton)
env = gym.make('myoLegWalk-v0', 
               model_path='models/22muscle_2D/myoLeg22_2D_BASELINE.xml')

# Reset environment
obs, info = env.reset()

# Run a simple simulation loop
for i in range(1000):
    # Random action (muscle activations)
    action = env.action_space.sample()
    
    # Step simulation
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Render (visualize)
    env.render()
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()