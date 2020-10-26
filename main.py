import random
import gym
import numpy as np
import statistics
from collections import deque
from keras.models import Sequential
from tensorflow.keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam


class Agent:
    def __init__(self,state_size,action_size,memory_size=50000,batch_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.discount_factor = 0.95 # gamma
        self.learning_rate = 0.002  # alpha
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99975
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24,input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        # do this before prediction
        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # do batch prediction to save speed
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)

        for i in range(self.batch_size):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # Q_max = max_a' Q_target(s', a')
                target[i][action[i]] = reward[i] + self.discount_factor * (np.amax(target_next[i]))

        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def getEpsilon(self):
        return self.epsilon

    def save(self, name):
        self.model.save(name)

    def load(self, name):
        self.model = load_model(name)


def test(agent):
    agent.load("cartpole-dqn.h5")
    episode = 0
    agent.epsilon = 0

    while True :
        total_reward = 0
        observation = env.reset()
        state = np.reshape(observation, [1, state_size])
        episode += 1
        # we test if our agent is able to perform 200 frames
        for i in range(200):
            env.render()
            total_reward += 1
            action = agent.act(state)
            observation, reward, done, _ = env.step(action)
            state = np.reshape(observation, [1, state_size])
            if done :
                print("Episode %d/ finished score : %f."
                      % (episode,  total_reward))
                break



def train(agent):
    score100_mean = 0
    episode = 0
    while score100_mean < 200:
        episode = episode + 1
        # Defines the total reward per episode
        total_reward = 0

        # Resets the environment
        observation = env.reset()

        # Gets the state
        state = np.reshape(observation, [1, state_size])

        while True:
            # Renders the screen after new environment observation
            #env.render()

            # Gets a new action
            action = agent.act(state)

            # Takes action and calculates the total reward
            observation, reward, done, _ = env.step(action)
            total_reward += reward

            # Gets the next state
            next_state = np.reshape(observation, [1, state_size])

            # Memorizes the experience
            agent.remember(state, action, reward, next_state, done)

            # Updates the state
            state = next_state

            # Updates the network weights
            agent.replay()

            if done:
                score100_list.append(total_reward)
                score100_mean = statistics.mean(score100_list)
                print("Episode %d finished with total reward = %f.  epsilon = %f. score100 : %f."
                      % (episode + 1,  total_reward, agent.getEpsilon(), score100_mean))


                break

    agent.save("cartpole-dqn.h5")


if __name__ == "__main__":
    #init env
    env = gym.make('CartPole-v1')

    ###PARAM
    score100_list = deque(maxlen=100)
    ###

    state_size = env.observation_space.shape[0]
    # States of the cart : 4
    """ Observation: 
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24°           24°
        3	Pole Velocity At Tip      -Inf            Inf    """

    action_size = env.action_space.n
    # Action of the cart : 2
    """ Action:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right """

    agent = Agent(state_size, action_size)

    train(agent)

    #test(agent)



