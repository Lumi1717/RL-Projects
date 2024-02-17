import gym
import random
# import tensorflow 
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam 
import numpy as np


env = gym.make('CartPole-v0')
states = env.observation_space.shape[0]
actions = env.action_space.n

episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0


    while not done:
        env.render()
        action = random.choice([0,1])
        n_state, reward, done, info = env.step(action=action)
        score += reward
    print('Episode: {0} Score: {1}'.format(episode, score))


# Create Deep Learning Model with Keras
def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


model = build_model(states=states, actions=actions)
model.summary()


# Build agent with Keras-RL

from rl.agents import DQNAgent 
from rl.policy import BoltzmannGumbelQPolicy
from rl.memory import SequentialMemory

