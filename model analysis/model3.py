import os
import random
import numpy as np
import gym
import keras
import copy
from keras.models import load_model
import time


model_name = 'fittest_brain3.h5'
if os.path.exists(model_name):
    os.remove(model_name)


class Agent():
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.brain = self.create_brain()
        self.memory = []
        self.exploration_rate = 0.1
        self.min_exloration_rate = 0.01
        self.exploration_rate_reduction = 0.1
        self.learning_rate = 0.005
        self.discount_factor = 0.9
        self.losing_penalty = 3


    def create_brain(self):
        inputs = keras.layers.Input(shape=(4,))
        x = keras.layers.Dense(32, activation='linear')(inputs)
        x = keras.layers.LeakyReLU(alpha=0.3)(x)
        x = keras.layers.Dense(64, activation='linear')(x)
        x = keras.layers.LeakyReLU(alpha=0.3)(x)
        predictions = keras.layers.Dense(2, activation='linear')(x)
        model = keras.Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer='rmsprop', loss='mean_squared_error')
        return model


    def play(self, num_episodes=10, num_time_steps=150):
        self.memory = []
        for episode in range(num_episodes):
            observation = self.env.reset()
            for t in range(num_time_steps):
                initial_observation = observation.reshape(1, -1)
                initial_q_values = self.brain.predict(initial_observation).flatten()
                if np.random.rand() < self.exploration_rate:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(initial_q_values)
                observation, reward, done, _ = self.env.step(action)
                observation = observation.reshape(1, -1)
                q_values = self.brain.predict(observation).flatten()
                if done:
                    target = copy.copy(initial_q_values)
                    target[action] = initial_q_values[action] + self.learning_rate * (initial_q_values[action] - self.losing_penalty)
                    self.memory.append((initial_observation, initial_q_values, observation, q_values, target, action, reward, done))
                    break
                else:
                    target = copy.copy(initial_q_values)
                    target[action] = initial_q_values[action] + self.learning_rate * (initial_q_values[action] - (reward + self.discount_factor * np.max(q_values)))
                    self.memory.append((initial_observation, initial_q_values, observation, q_values, target, action, reward, done))


    def learn_from_memory(self, batch_size=10, num_epochs=1):
        self.memory = np.asarray(self.memory)
        x = self.memory[:, 0]
        y = self.memory[:, 4]
        x = np.concatenate(x)
        y = np.concatenate(y).reshape(-1, 2)
        for epoch in range(num_epochs):
            self.brain.fit(x, y, batch_size=batch_size, shuffle=True)


    def train(self, num_training_sessions=1000):
        max_brain_size = np.asarray(self.memory).size
        for training_session in range(num_training_sessions):
            self.play()
            if np.asarray(self.memory).size > max_brain_size:
                self.brain.save(model_name)
                max_brain_size = np.asarray(self.memory).size
            self.learn_from_memory()
            self.exploration_rate = max(self.min_exloration_rate, self.exploration_rate * self.exploration_rate_reduction)


    def showcase(self, num_episodes=50, num_time_steps=150):
        self.brain = load_model(model_name)
        average_time_steps = 0
        for episode in range(num_episodes):
            observation = self.env.reset()
            for t in range(num_time_steps):
                # self.env.render()
                initial_observation = observation.reshape(1, -1)
                initial_q_values = self.brain.predict(initial_observation).flatten()
                action = np.argmax(initial_q_values)
                observation, reward, done, _ = self.env.step(action)
                observation = observation.reshape(1, -1)
                if t == num_time_steps - 1:
                    average_time_steps = average_time_steps + 1 / (episode + 1) * (t - average_time_steps)
                    print('The average number of steps is %s' %(average_time_steps))
                if done:
                    average_time_steps = average_time_steps + 1 / (episode + 1) * (t - average_time_steps)
                    print('The average number of steps is %s' %(average_time_steps))
                    break


agent = Agent()
start_time = time.time()
agent.train()
print('Training took ', time.time() - start_time, "to run")
agent.showcase()
