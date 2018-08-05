from collections import deque
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import numpy as np
import random

# https://keon.io/deep-q-learning/
# Deep Q-learning Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2500)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)[0]
        act_values -= np.min(act_values)
        act_values /= np.sum(act_values)
#        a = np.random.choice(len(act_values), 1, True, act_values)
#        return a
        return np.argmax(act_values)
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        loss = 0
        
        # TODO: This is not really a 'minibatch'
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            hist = self.model.fit(state, target_f, epochs=1, verbose=0)
            loss += hist.history['loss'][0]
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss / float(batch_size)