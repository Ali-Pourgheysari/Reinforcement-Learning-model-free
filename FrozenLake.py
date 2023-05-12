# import library
import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# create Enviroment
size = 16
env = gym.make("FrozenLake-v1", desc=generate_random_map(size=size),
               render_mode="human", is_slippery=True)

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

# set the reward of each state
for i, rowVal in enumerate(env.unwrapped.spec.kwargs['desc']):
    for j, colVal in enumerate(rowVal):
        coord = i*size+j
        if colVal == "H":
            Rewards[coord] = hole_reward
        elif colVal == "G":
            Rewards[coord] = goal_reward
        else:
            Rewards[coord] = state_reward

# calculate if the agent can move to the next state


def CanMoveToNextState(observation, action):
    if env.unwrapped.P[observation][action][0][1] == observation:
        return False
    else:
        return True

# choose action


def choose_action():
    actions_index = []

    # Get the index and value of each action
    for i in range(4):
        actions_index.append((QTable[observation][i], i))

    # Sort the actions by their values
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

    # Choose the action with the highest value with probability (1 - epsilon) (epsilon-greedy)
    if np.random.uniform(0, 1) < (1 - epsilon):
        action = shuffled_list[0][1]
    else:
        action = env.action_space.sample()

    # If the action is not valid, choose the next action with the highest value
    i = 1
    while True:
        if CanMoveToNextState(observation, action):
            break
        action = sorted_list[i][1]
        i += 1

    # Return the chosen action
    return action

# Q-learning algorithm
def Q_learning():

    while True:
        action = choose_action()
        new_observation, reward, terminated, truncated, info = env.step(action)

        # get the maximum value of the next state
        available_actions = [value for i, value in enumerate(
            QTable[new_observation]) if CanMoveToNextState(new_observation, i)]

        # if the maximum list is empty, it means that the agent is in a hole
        # so we add the hole reward to the maximum list and decrease the hole reward
        # so next time the agent will not get stock in the hole
        if not available_actions:
            global hole_reward
            hole_reward -= size
            available_actions.append(hole_reward)

        # update the QTable
        global observation
        QTable[observation, action] = QTable[observation, action] + alpha * \
            (Rewards[new_observation] + gamma *
             np.max(available_actions) - QTable[observation, action])

        # update the Rewards
        Rewards[observation] = np.average(QTable[observation])

        # print the actions of the old state, the actions of the new state, 
        # the action chosen, the reward of the old state and the reward of the new state
        print(QTable[observation], QTable[new_observation],
              action, Rewards[observation], Rewards[new_observation])
        observation = new_observation

        if terminated or truncated:
            observation, info = env.reset()
            break

# learning phase
for _ in range(max_iter_number):
    Q_learning()

# test phase
while True:
    action = np.argmax(QTable[observation])
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
