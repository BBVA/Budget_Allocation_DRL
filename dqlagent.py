
import os
import random
import warnings
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp

from collections import deque
from tqdm import tqdm
import time

class DQLAgent:
    def __init__(self, symbol, feature, n_features, env, hu=24, lr=0.001):
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.5
        self.trewards = list()
        self.max_treward = -np.inf
        self.n_features = n_features
        self.env = env
        self.episodes = 0
        self._create_model(hu, lr)
        
    def _create_model(self, hu, lr):
        self.model = Sequential()
        self.model.add(Dense(hu, activation='relu',
                             input_dim=self.n_features))
        self.model.add(Dense(hu, activation='relu'))
        self.model.add(Dense(2, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=lr))
        
    def _reshape(self, state):
        state = state.flatten()
        return np.reshape(state, [1, len(state)])
            
    def act(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])
        
    def replay(self):
        batch = random.sample(self.memory, self.batch_size)
        for state, action, next_state, reward, done in batch:
            if not done:
                reward += self.gamma * np.amax(
                    self.model.predict(next_state)[0])
                target = self.model.predict(state)
                target[0, action] = reward
                self.model.fit(state, target, epochs=1, verbose=False)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def learn(self, episodes):
        for e in tqdm(range(1, episodes + 1)):
            self.episodes += 1
            state, _ = self.env.reset()
            state = self._reshape(state)
            treward = 0

            #start = time.process_time()
            self.memory.clear()
            for f in range(1, 5000):
                self.f = f
                action = self.act(state)
                                                
                next_state, reward, done, trunc, _ = self.env.step(action)
                
                treward += reward
                next_state = self._reshape(next_state)
                self.memory.append([state, action, next_state, reward, done])
                state = next_state 
                if done:
                    self.trewards.append(treward)
                    self.max_treward = max(self.max_treward, treward)
                    templ = f'episode={self.episodes:4d} | '
                    templ += f'epsilon={self.epsilon:7.3f} | '
                    templ += f'treward={treward:7.3f}'
                    templ += f' | max={self.max_treward:7.3f}'
                    print(templ, end='\n')
                    break
            if len(self.memory) > self.batch_size:
                self.replay()  
        
