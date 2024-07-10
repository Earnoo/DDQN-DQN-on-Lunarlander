# DDQN Lunar Lander

Project Goal:
------------

The goal of this project is to train an AI agent to achieve an average score of over 200 points per episode in the Lunar Lander game using reinforcement learning techniques.

Environment:

https://github.com/Earnoo/DDQN-DQN-on-Lunarlander/assets/134099561/01ae160e-9e3f-40f3-8072-52354a722a16

------------
The Lunar Lander environment from OpenAI Gym is utilized. It consists of a continuous state space with 8 dimensions:
(x, y, v_x, v_y, theta, v_theta, leg_left, leg_right)

Actions:
--------
There are 4 discrete actions available:
1. Do nothing
2. Fire left orientation engine
3. Fire main engine
4. Fire right orientation engine

Scoring:
--------
- Landing on the landing pad: +100 points
- Crashing or going out of bounds: -100 points
- Each main engine firing (action 3): -0.3 points
- Each leg contact with the ground: +10 points

## Deep Q-Network
Deep Q-Network (DQN) is a reinforcement learning method that utilizes deep neural networks to approximate the Q-function. The Q-function represents the expected total reward value from taking action $a$ in state $s$ and following the optimal policy thereafter.

In DQN, a deep neural network is used as an approximator to calculate Q-values for state-action pairs.

The Q-function update equation in DQN is given by:

Q(s_t, a_t) <- Q(s_t, a_t) + α [r_t + γ max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)]

Where:
- $s_t$: State at time $t$
- $a_t$: Action taken at time $t$
- $r_t$: Reward received at time $t$
- $s_{t+1}$: New state after taking action $a_t$
- $\alpha$: Learning rate
- $\gamma$: Discount factor

DQN employs Experience Replay and a Target Network to improve learning stability and efficiency. In Experience Replay, agent experiences are stored in a memory buffer and sampled randomly to reduce correlations between experiences. The Target Network is a copy of the main network that is periodically updated episodically to mitigate large fluctuations during learning.

## Double Deep Q-Network
Double Deep Q-Network (DDQN) is an advanced version of the Deep Q-Network (DQN) that addresses issues of stability and learning accuracy in reinforcement learning. The primary issue with DQN is overestimation of Q-values due to using the same network for action selection and evaluation. DDQN mitigates this problem by separating these two tasks into two distinct networks.

In DDQN, two neural networks are employed:
- Policy Network or Main Network: Used for action selection.
- Target Network: Used for evaluating Q-values.

The update equations in DDQN are as follows:

### Action Selection

Initially, actions are chosen using the Policy Network:

a* = arg max_a Q(s, a; θ)

### Computing Target Q-Value

Subsequently, the target Q-value is computed using the Target Network:

Q_target = r + γ Q(s', a*; θ^-)

Where:
- r: Received reward
- γ: Discount factor
- s': Next state
- θ: Parameters of the Policy Network
- θ^-: Parameters of the Target Network

### Updating Policy Network Parameters

The mean squared error between the target Q-values and predicted Q-values by the Policy Network is computed:

loss = (1 / N) Σ_{i=1}^N (Q_target - Q(s, a; θ))^2

The parameters of the Policy Network are updated using an optimization algorithm (such as Adam).

### Comparing DDQN and DQN

The main difference between DDQN and DQN lies in how the target Q-value is computed. In DQN, the target Q-value is computed as:

Q_target = r + γ max_{a'} Q(s', a'; θ)

This leads to Q-values being overestimated due to using the same network for action selection and evaluation. In contrast, DDQN uses two separate networks for these tasks, which helps reduce overestimation.

In summary:
- DQN uses one network for both action selection and evaluation, leading to overestimation.
- DDQN uses two separate networks, which separates action selection and evaluation tasks and helps reduce overestimation.

This improvement allows DDQN to exhibit greater stability in the learning process and achieve better performance in various reinforcement learning tasks.

The following code implements the Double Deep Q-Network (DDQN), introduced by Hasselt et al. in 2016. This code includes three main sections: the neural network model, experience replay memory, and the DDQN agent. Each section of the code is detailed further below.


## Dependencies
+ python >= 3.7.2
+ jupyter >= 1.0.0
+ numpy>=1.16.2
+ gym >= 0.16.0
+ torch >= 1.4.0
+ tqdm >= 4.43.0

## Setup
Please ensure the following packages are already installed. A virtual environment is recommended.
+ Python (for .py)
+ Jupyter Notebook (for .ipynb)

```
$ cd DDQN-Lunar-Lander/
$ pip3 install pip --upgrade
$ pip3 install -r requirements.txt
```

## Output

## Results
### Learning episodes
#### Episode 50

https://github.com/Earnoo/DDQN-DQN-on-Lunarlander/assets/134099561/ccfc2f08-6b05-496b-b917-c3f57fa4296a

#### Episode 100
https://github.com/Earnoo/DDQN-DQN-on-Lunarlander/assets/134099561/7fe1b6ab-95d1-44fd-89bc-2c18e8517e03

#### Episode 250

https://github.com/Earnoo/DDQN-DQN-on-Lunarlander/assets/134099561/2e1a6616-7add-4841-bce8-b9d2bc8b56d0

#### Episode 350

https://github.com/Earnoo/DDQN-DQN-on-Lunarlander/assets/134099561/800e142d-8c1d-4319-98a4-fbd192cc6c92

### DQN
https://github.com/Earnoo/DDQN-DQN-on-Lunarlander/assets/134099561/5d811f87-e87d-4b4d-b6bf-ac7f281adf8e
### DDQN Agent
https://github.com/Earnoo/DDQN-DQN-on-Lunarlander/assets/134099561/7f02f4de-dec3-43f8-b9d0-b83094737411

## Reference
1. Hasselt, H. V. (2010). Double Q-learning. In Advances in neural information processing systems (pp. 2613-2621).
2. Van Hasselt, H., Guez, A., & Silver, D. (2016, March). Deep reinforcement learning with double q-learning. In Thirtieth AAAI conference on artificial intelligence.
3. Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J., Tang, J., & Zaremba, W. (2016). Openai gym. arXiv preprint arXiv:1606.01540.
4. Fujimoto, S., Van Hoof, H., & Meger, D. (2018). Addressing function approximation error in actor-critic methods. arXiv preprint arXiv:1802.09477.
5. (https://mrshininnnnn.github.io/)
