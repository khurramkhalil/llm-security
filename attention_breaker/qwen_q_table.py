import random
import numpy as np
from collections import defaultdict

class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.99):
        self.q_table = defaultdict(lambda: np.zeros(action_space))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(range(len(state)))
        else:
            q_values = self.q_table[tuple(state)]
            return np.argmax(q_values)

    def learn(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[tuple(next_state)])
        td_target = reward + self.discount_factor * self.q_table[tuple(next_state)][best_next_action]
        td_error = td_target - self.q_table[tuple(state)][action]
        self.q_table[tuple(state)][action] += self.learning_rate * td_error
        self.exploration_rate *= self.exploration_decay

def train_rl_agent(env, episodes=1000):
    agent = QLearningAgent(action_space=len(env.top_k_indices) * 32)
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
        print(f"Episode {episode + 1}/{episodes} completed")
    return agent