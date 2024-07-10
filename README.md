# DDQN Lunar Lander

Project Goal:
------------
The goal of this project is to train an AI agent to achieve an average score of over 200 points per episode in the Lunar Lander game using reinforcement learning techniques.

Environment:
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

Implementation:
---------------
The agent is implemented using a Deep Q-Network (DQN) algorithm in Python with TensorFlow and OpenAI Gym. The project includes code for training the agent to learn optimal policies for landing the Lunar Lander.

For more details and code examples, please refer to the accompanying Python scripts.



## Double Deep Q-Network
In 2010, Hasselt found an overestimation bias when using Q-Learning to solve several Atari games. Taking a special state as an example, where the real Q value for every action equals zero, the estimated Q values will be around zero and cause the overestimation of Q values. There is a big problem with learning estimates from estimates in the Q-Learning method. Fortunately, it can be tackled by employing two separate Q-value estimators updating each other. Thus, this approach has the name Double Q-Learning, which is one of the most popular methods in recent years (Hasselt 2010; Van Hasselt, Guez, and Silver 2016; Fujimoto, van Hoof, and Meger 2018). Borrowing the idea of Hasselt et al. (2016), we applied the Double Deep Q-Network (DDQN) with the assistance of deep learning.

## Directory
+ **main.py** - to conduct the entire project directly and show some figures
+ **main.ipynb** - to go through the modeling, training, and evaluation step by step
+ **config.py** - to set the configuration for model development and training pipeline
```
DDQN-Lunar-Lander/
├── README.md
├── config.py
├── main.ipynb
├── main.py
├── output
│   ├── ddqn_agent
│   └── random_agent
├── reference
├── requirements.txt
├── res
└── src
    ├── agent.py
    └── model.py
```

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

To generate video record for an episode: 
* On OS X, you can install ffmpeg via `brew install ffmpeg`. 
* On most Ubuntu variants, `sudo apt-get install ffmpeg` should do it. 
* On Ubuntu 14.04, however, you'll need to install avconv with `sudo apt-get install libav-tools`.

## Output
```
Initialize the environment...



## Results

### Random Agent
<p align="center"><img src="./res/Random_Agent.gif" alt="Random Agent"  width="320" /></p>

### DDQN Agent
<p align="center"><img src="./output/ddqn_agent/result_img_0.png" alt="Result 0"  width="405" /> <img src="./output/ddqn_agent/result_img_1.png" alt="Result 1"  width="405" /></p>
<p align="center"><img src="./res/DDQN_Agent.gif" alt="DDQN Agent"  width="320" /></p>

## Authors
* **[Ning Shi](https://mrshininnnnn.github.io/)** - MrShininnnnn@gmail.com

## Reference
1. Hasselt, H. V. (2010). Double Q-learning. In Advances in neural information processing systems (pp. 2613-2621).
2. Van Hasselt, H., Guez, A., & Silver, D. (2016, March). Deep reinforcement learning with double q-learning. In Thirtieth AAAI conference on artificial intelligence.
3. Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J., Tang, J., & Zaremba, W. (2016). Openai gym. arXiv preprint arXiv:1606.01540.
4. Fujimoto, S., Van Hoof, H., & Meger, D. (2018). Addressing function approximation error in actor-critic methods. arXiv preprint arXiv:1802.09477.
