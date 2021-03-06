import os
import numpy as np
import gym
import keras
import copy
from keras.models import load_model
import time

# Delete the existing model
model_name = 'test_model.h5'
if os.path.exists(model_name):
    os.remove(model_name)


class Agent():
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.brain = self.create_brain()
        self.memory = []
        self.exploration_rate = 0.3
        self.min_exloration_rate = 0.01
        self.exploration_rate_reduction = 0.8
        self.learning_rate = 0.0001
        self.discount_factor = 1
        self.losing_penalty = -100

    # Create the model
    def create_brain(self):
        inputs = keras.layers.Input(shape=(4,))
        x = keras.layers.Dense(128, activation='linear')(inputs)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.LeakyReLU(alpha=0.3)(x)
        x = keras.layers.Dense(256, activation='linear')(x)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.LeakyReLU(alpha=0.3)(x)
        predictions = keras.layers.Dense(2, activation='linear')(x)
        model = keras.Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    # Navigate through the environment and collect traning data in memory
    def play(self, num_episodes=10, num_time_steps=250):
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
                    target_test = copy.copy(initial_q_values)
                    target[action] = initial_q_values[action] + self.learning_rate * (self.losing_penalty - initial_q_values[action])
                    target_test[action] = initial_q_values[action] + self.learning_rate * (initial_q_values[action] + self.losing_penalty)
                    self.memory.append((initial_observation, initial_q_values,  q_values, target, target_test, action, reward, done))
                    break
                else:
                    target = copy.copy(initial_q_values)
                    target_test = copy.copy(initial_q_values)
                    target[action] = initial_q_values[action] + self.learning_rate * (reward + self.discount_factor * np.max(q_values) - initial_q_values[action])
                    target_test[action] = initial_q_values[action] + self.learning_rate * (initial_q_values[action] - (reward + self.discount_factor * np.max(q_values)))
                    self.memory.append((initial_observation, initial_q_values, q_values, target, target_test, action, reward, done))

    # Used collected data from the environment to learn
    def learn_from_memory(self, batch_size=10, num_epochs=1):
        self.memory = np.asarray(self.memory)
        x = self.memory[:, 0]
        y = self.memory[:, 4]
        x = np.concatenate(x)
        y = np.concatenate(y).reshape(-1, 2)
        for epoch in range(num_epochs):
            self.brain.fit(x, y, batch_size=batch_size, shuffle=True)

    # Repeat the play/learn process for a specified number of training sessions
    def train(self, num_training_sessions=2000):
        max_brain_size = np.asarray(self.memory).size
        for training_session in range(num_training_sessions):
            self.play()
            if np.asarray(self.memory).size > max_brain_size:
                self.brain.save(model_name)
                max_brain_size = np.asarray(self.memory).size
            self.learn_from_memory()
            self.exploration_rate = max(self.min_exloration_rate, self.exploration_rate * self.exploration_rate_reduction)

    # Test the actual performance
    def showcase(self, num_episodes=100, num_time_steps=250):
        self.brain = load_model(model_name)
        average_time_steps = 0
        for episode in range(num_episodes):
            observation = self.env.reset()
            for t in range(num_time_steps):
                self.env.render()
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
print('Training took ', time.time() - start_time, "seconds to run")
agent.showcase()
