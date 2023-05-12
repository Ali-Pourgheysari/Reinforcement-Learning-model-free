# import library
import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# create Enviroment
size = 16
env = gym.make("FrozenLake-v1", desc=generate_random_map(size=size),
               render_mode="human", is_slippery=False)

observation, info = env.reset(seed=42)

# define parameters
max_iter_number = 1000
alpha = 0.3
gamma = 0.5
epsilon = 0.1
hole_reward = -1 * size
goal_reward = 10**size
state_reward = -1

# create QTable
QTable = np.zeros((size*size, 4))
# create Rewards
Rewards = np.zeros(size*size)

# set the reward of the goal and the hole
for i, rowVal in enumerate(env.unwrapped.spec.kwargs['desc']):
    for j, colVal in enumerate(rowVal):
        coord = i*size+j
        if colVal == "H":
            Rewards[coord] = hole_reward
        elif colVal == "G":
            Rewards[coord] = goal_reward
        else:
            Rewards[coord] = state_reward


def CanMoveToNextState(observation, action):
    if env.unwrapped.P[observation][action][0][1] == observation:
        return False
    else:
        return True


def choose_action():
    actions_index = []
    for i in range(4):
        actions_index.append((QTable[observation][i], i))

    sorted_list = sorted(actions_index, key=lambda x: x[0], reverse=True)

    # Initialize a dictionary to group elements by their first values
    groups = {}

    # Iterate through the sorted list and group elements by their first values
    for elem in sorted_list:
        key = elem[0]
        if key not in groups:
            groups[key] = []
        groups[key].append(elem)

        # Shuffle the elements within each group
        for key in groups:
            np.random.shuffle(groups[key])

    # Flatten the groups back into a single list
    shuffled_list = [elem for group in groups.values() for elem in group]

    # shuffled_list[0][1] > 0 or
    if np.random.uniform(0, 1) < (1 - epsilon):
        action = shuffled_list[0][1]
    else:
        action = env.action_space.sample()

    i = 1
    while True:
        if CanMoveToNextState(observation, action):
            break
        action = sorted_list[i][1]
        i += 1

    return action


def Q_learning():

    while True:
        action = choose_action()
        new_observation, reward, terminated, truncated, info = env.step(action)

        maximum_list = [value for i, value in enumerate(
            QTable[new_observation]) if CanMoveToNextState(new_observation, i)]

        if not maximum_list:
            global hole_reward
            maximum_list.append(hole_reward)
            hole_reward -= size

        global observation
        QTable[observation, action] = QTable[observation, action] + alpha * \
            (Rewards[new_observation] + gamma *
             np.max(maximum_list) - QTable[observation, action])

        Rewards[observation] = np.average(QTable[observation])

        print(QTable[observation], QTable[new_observation],
              action, Rewards[observation], Rewards[new_observation])
        observation = new_observation

        if terminated or truncated:
            observation, info = env.reset()
            break


for _ in range(max_iter_number):
    Q_learning()

env.close()
