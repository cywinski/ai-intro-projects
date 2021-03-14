### NAGRODY ###
# Duza dodatnia nagroda za dojscie do sera
# -1 Za kazdy ruch
# Duza ujemna nagroda za wejscie na dziure

### STANY ###
# n x n - kazda pozycja labiryntu

### AKCJE ###
# 0 - lewo
# 1 - prawo
# 2 - gora
# 3 - dol


import numpy as np
import random
class Q_Learning():

    def __init__(self, start, end, maze, epsilon, max_epsilon, min_epsilon, decay_rate, beta, gamma, total_epochs, max_steps, min_paths):
        self.start = start
        self.end = end
        self.maze = maze
        self.epsilon = epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.beta = beta
        self.gamma = gamma
        self.total_epochs = total_epochs
        self.max_steps = max_steps
        self.Qtable = np.zeros((maze.get_n()**2, 4))
        self.min_paths = min_paths
        self.learn()

    ## KODOWANIE STANOW - Q[0]*4 + Q[1]
    def get_Q_row(self, x, y):
        return x*4 + y

    # Wylosowanie akcji
    # Strategia epsilon - zachlanna
    # Z prawdopodobienstwem = epsilon wybieram losowa akcje
    # Z prawdopodobienstwem = 1 - epsilon wybieram najlepszą akcje (najwyzsza wartosc w rzedzie)
    def pick_action(self, state):
        return np.random.choice(a=[random.randint(0, 3), np.argmax(self.Qtable[self.get_Q_row(state[0], state[1])])], p=[self.epsilon, 1-self.epsilon])


    # Wykonaj akcje, zarejestruj kolejny stan oraz nagrode
    def make_action(self, state, action):
        reward = 0
        next_state = None
        # lewo
        if action == 0:
            next_state = (state[0], state[1] - 1)
        # prawo
        elif action == 1:
            next_state = (state[0], state[1] + 1)
        # gora
        elif action == 2:
            next_state = (state[0] - 1, state[1])
        # dol
        elif action == 3:
            next_state = (state[0] + 1, state[1])

        # stan jest poza labiryntem lub jest dziura
        if not self.maze.in_matrix(next_state[0], next_state[1]) or self.maze.get_matrix()[next_state[0], next_state[1]] == 1:
            reward = -10
        # stan jest miejscem, gdzie znajduje sie ser
        elif next_state == self.end:
            reward = 20
        else:
            reward = -1

        return next_state, reward

    def improve_Q(self, state, action, reward, next_state):
        self.Qtable[self.get_Q_row(state[0], state[1]), action] += \
            self.beta * (reward + self.gamma * np.max(self.Qtable[self.get_Q_row(next_state[0], next_state[1]), :]) \
            - self.Qtable[self.get_Q_row(state[0], state[1]), action])

    def learn(self):
        #BRAIN
        brain_end = 0
        brain_min_paths = 0
        brain_paths_len = list()

        epoch = 1
        while epoch <= self.total_epochs:
            t = 1
            curr_state = self.start
            #print(f"BRAIN epoch: {epoch}, step: {t-1}, state: {curr_state}")
            path = [self.start]
            while t <= self.max_steps:
                
                action = self.pick_action(curr_state)
                next_state, reward = self.make_action(curr_state, action)
                if not self.maze.in_matrix(next_state[0], next_state[1]):
                    next_state = curr_state
                self.improve_Q(curr_state, action, reward, next_state)
                if self.maze.in_matrix(next_state[0], next_state[1]):
                    curr_state = next_state
                #print(f"BRAIN epoch: {epoch}, step: {t}, state: {curr_state}")
                path.append(curr_state)
                if self.maze.is_terminal(curr_state[0], curr_state[1]):
                    brain_paths_len.append(len(path))
                    if self.end in path:
                        brain_end += 1
                    if path in self.min_paths:
                        brain_min_paths += 1
                    break
                
                t += 1
            epoch += 1
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon)*np.exp(-self.decay_rate * epoch)

        # PINKY
        pinky_end = 0
        pinky_min_paths = 0
        pinky_paths_len = list()
        epoch = 1
        while epoch <= self.total_epochs:
            t = 1
            curr_state = self.start
            #print(f"PINKY epoch: {epoch}, step: {t-1}, state: {curr_state}")
            path = [self.start]
            while t <= self.max_steps:
                
                action = random.randint(0, 3)
                next_state, reward = self.make_action(curr_state, action)
                if not self.maze.in_matrix(next_state[0], next_state[1]):
                    next_state = curr_state
                else:
                    curr_state = next_state

                #print(f"PINKY epoch: {epoch}, step: {t}, state: {curr_state}")
                path.append(curr_state)
                if self.maze.is_terminal(curr_state[0], curr_state[1]):
                    pinky_paths_len.append(len(path))
                    if self.end in path:
                        pinky_end += 1
                    if path in self.min_paths:
                        pinky_min_paths += 1
                    break
                
                t += 1
            epoch += 1

        return brain_end, brain_min_paths, brain_paths_len, \
               pinky_end, pinky_min_paths, pinky_paths_len
