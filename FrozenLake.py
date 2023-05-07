# import library
import numpy as np
import gymnasium as gym

from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# create Enviroment
size = 16
env = gym.make("FrozenLake-v1", desc=generate_random_map(size=size),
               render_mode="human", is_slippery=False)

observation, info = env.reset(seed=42)

max_iter_number = 1000
alpha = 0.5
gamma = 0.9
epsilon = 0.1

QTable = np.zeros((size*size, 4))
for i in range(size*size):
    if (env.unwrapped.P[i][0][0][1] == i) and (env.unwrapped.P[i][0][0][3] == True):
        QTable[i] = -1 * np.ones(4)
QTable[size*size-1] = np.ones(4)

for _ in range(max_iter_number):

    max_val = np.max(QTable[observation])
    max_val_index = np.where(QTable[observation] == max_val)

    action = np.random.choice(max_val_index[0])

    if np.random.uniform() > epsilon:
        
        if 1 in max_val_index[0] and 2 in max_val_index[0]:
            action = np.random.choice([1, 2])
        elif 1 in max_val_index[0]:
            action = 1
        elif 2 in max_val_index[0]:
            action = 2 

    new_observation, reward, terminated, truncated, info = env.step(action)

    QTable[observation, action] = QTable[observation, action] + alpha * \
        (reward + gamma *
         np.max(QTable[new_observation]) - QTable[observation, action])

    observation = new_observation

    if terminated or truncated:

        observation, info = env.reset()


env.close()
