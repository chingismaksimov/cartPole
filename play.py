import gym
from keras.models import load_model
import numpy as np

env = gym.make("CartPole-v0")
agent = load_model("fittest_model.h5")

while True:
    observation = env.reset()
    for t in range(250):
        env.render()
        action = np.argmax(agent.predict(observation.reshape(1, -1)))
        # action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)
        if done:
            break
