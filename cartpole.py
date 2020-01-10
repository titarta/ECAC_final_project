import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"

RUN_NAME = "0.25SupRate"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
MEMORY_SIZE_DQN = 10000
BATCH_SIZE_PREDICTOR = 64
BATCH_SIZE = 64

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.96

SUPERVISION_RATE = 0.25

NUMBER_EPISODES = 100

class Reward_predictor:

    def __init__(self, input_space, output_space):

        self.output_space = output_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(6, input_shape=(input_space,), activation="relu"))
        self.model.add(Dense(4, activation="relu"))
        self.model.add(Dense(self.output_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))
        self.firstFit = False

    def remember(self, state_next, reward):
        self.memory.append((state_next, reward))

    def predict(self, state):
        q_values = self.model.predict(state)
        return q_values[0,0]

    def batch_fit(self):
        if len(self.memory) < BATCH_SIZE:
            return
        self.firstFit = True
        batch = random.sample(self.memory, BATCH_SIZE)
        for state_next, reward in batch:
            self.model.fit(state_next, [reward], verbose=0)

class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE_DQN)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        
    def exploration_decay(self):
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)
    


def cartpole():
    env = gym.make(ENV_NAME)
    score_logger = ScoreLogger(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    reward_predictor = Reward_predictor(observation_space, 1)
    run = 0
    scores = []
    for i in range(NUMBER_EPISODES):
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            step += 1
            #env.render()
            action = dqn_solver.act(state) # gets action here <--------DQN
            state_next, reward, terminal, info = env.step(action)
            reward = 1/(abs(state_next[2])+0.1) # reward function here <----
            reward = reward if not terminal else -100
            state_next = np.reshape(state_next, [1, observation_space])

            #doesn't consider reward
            if random.uniform(0,1) > SUPERVISION_RATE:
                reward = reward_predictor.predict(state_next)
                if terminal:
                    print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                    break
                if(not reward_predictor.firstFit):
                    state = state_next
                    continue
            else:
                reward_predictor.remember(state_next, reward)
                reward_predictor.batch_fit()
                

            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                #score_logger.add_score(step, run)
                break
            dqn_solver.experience_replay()
        dqn_solver.exploration_decay()
        scores.append(step)
    plt.plot(scores)
    plt.savefig("results/{}.png".format(RUN_NAME))
    scores = np.array(scores)
    print("average score: {}".format(np.mean(scores)))
    print("average score last 20: {}".format(np.mean(scores[-20:])))
    np.save("results/{}".format(RUN_NAME),scores)
    


if __name__ == "__main__":
    cartpole()
