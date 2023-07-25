# Reinforcement Learning Projects
 This repository contains two Python scripts showcasing the implementation of Reinforcement Learning algorithms in the OpenAI Gym environment. The projects focus on solving two different environments, Frozen Lake and Taxi Driver, using Q-Learning algorithms.

## Part 1: Frozen Lake

### Description
The `frozen_lake.py` script demonstrates the application of Q-Learning to solve the Frozen Lake environment. The agent navigates through a grid world with slippery ice to reach the goal while avoiding holes. The Q-Table is updated using the Q-Learning algorithm, which gradually learns the optimal policy for maximizing rewards.

### Requirements
* Python 3.x
* OpenAI Gym library (gymnasium)

### How to Use
1. Install Dependencies: Make sure you have Python 3.x installed. Install the required library using the following command:
```
pip install gymnasium
```
2. Run the Script: Execute the script, and it will run Q-Learning to find the optimal policy for the Frozen Lake environment.

### Output
During the learning phase, the script prints the actions of the old state, the actions of the new state, the action chosen, the reward of the old state, and the reward of the new state for each iteration. The script updates the Q-Table and the Rewards matrix to optimize the agent's policy. After the learning phase, the script enters the test phase, where the agent follows the learned policy to reach the goal.

 Read the complete documentation [HERE](https://github.com/Ali-Pourgheysari/Reinforcement-Learning-model-based/Documentation.pdf).
Also you can read [THIS](FrozenLake_Code_review.pdf) code review for more details about the code.

## Part 2: Taxi Driver

### Description
The `taxi_driver.py` script demonstrates the application of Q-Learning to solve the Taxi Driver environment. The agent navigates through a grid world to pick up and drop off passengers at the correct locations. The Q-Table is updated using the Q-Learning algorithm, which gradually learns the optimal policy for maximizing rewards.

### Requirements
* Python 3.x
* OpenAI Gym library (gym)

### How to Use
1. Install Dependencies: Make sure you have Python 3.x installed. Install the required library using the following command:
```
pip install gym
```
2. Run the Script: Execute the script, and it will run Q-Learning to find the optimal policy for the Taxi Driver environment.

### Output
During the learning phase, the script updates the Q-Table to optimize the agent's policy. The script then enters the test phase, where the agent follows the learned policy to efficiently pick up and drop off passengers.

 Read the complete documentation [HERE](Taxi_Doc.pdf).
Also you can read [THIS](Taxi_Code_review.pdf) code review for more details about the code.

## Credits
The scripts in this repository were created as educational examples of applying Reinforcement Learning algorithms to solve different environments. The projects use the OpenAI Gym library, which provides a diverse collection of environments for Reinforcement Learning tasks. These projects can serve as a starting point for further exploration and understanding of reinforcement learning algorithms and their applications.

