import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

env = gym.make('CartPole-v1')
states = env.observation_space.shape[0]
actions = env.action_space.n

model = Sequential()
model.add(layers.Flatten(input_shape = (1, states)))
model.add(layers.Dense(24, activation = 'relu'))
model.add(layers.Dense(24, activation = 'relu'))
model.add(layers.Dense(actions, activation = 'linear'))

agent = DQNAgent(model = model,
               memory = SequentialMemory(limit = 50000, window_length = 1),
               policy = BoltzmannQPolicy(),
               nb_actions = actions,
               nb_steps_warmup = 10,
               target_model_update = 0.01)

agent.compile(tf.keras.optimizers.Adam(), metrics = ['mae'])
agent.fit(env, nb_steps = 100000, visualize = False, verbose = 1)

results = agent.test(env, nb_episodes = 10, visualize = True)
print(np.mean(results.history['episode_reward']))
env.close()