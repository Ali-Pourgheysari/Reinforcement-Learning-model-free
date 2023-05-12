import gym
import numpy as np
import pickle, os

# print(gym.__version__)


env = gym.make("Taxi-v3")

env.reset()

number_of_actions = env.action_space.n
number_of_states = env.observation_space.n


Q = np.zeros([number_of_states, number_of_actions])
reward = None  

# Train the agent
max_iter_number = 1000
G = 0 #goal state
alpha = 0.618

for episode in range(1,max_iter_number+1):
    isDone = False
    G, reward = 0,0 
    state = env.reset()

    while isDone != True:
        action = np.argmax(Q[state])
        nextState, reward, isDone, info = env.step(action)
        Q[state, action] += alpha * (reward + np.max(Q[nextState]) - Q[state, action]) 
        G += reward
        state = nextState
    


#finalState = state
#print(finalState)
#print(Q)

state = env.reset()
isDone = None

while isDone != True:
    action = np.argmax(Q[state])
    state, reward, isDone, info = env.step(action)
    env.render()

with open("TaxiProblem_Qtable.pkl", 'wb') as f:
    pickle.dump(Q, f)

# with open("TaxiProblem_Qtable.pkl", 'rb') as f:
#     QtestFile = pickle.load(f)

# print(QtestFile)