
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define a simple Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, action_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Hyperparameters
learning_rate = 0.001
gamma = 0.99 # Discount factor
epsilon = 1.0 # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01

# Environment
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Initialize Q-Network and optimizer
q_network = QNetwork(state_size, action_size)
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Training loop (simplified for demonstration)
def train_dqn(episodes=100):
    global epsilon
    for episode in range(episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        done = False
        total_reward = 0

        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q_network(state)
                    action = torch.argmax(q_values).item()

            next_state, reward, done, _, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            # Q-value update (simplified)
            with torch.no_grad():
                target_q_value = reward + gamma * torch.max(q_network(next_state)) * (1 - done)
            current_q_value = q_network(state)[0, action]

            loss = criterion(current_q_value, target_q_value)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            total_reward += reward

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Episode {episode+1}: Total Reward = {total_reward}, Epsilon = {epsilon:.2f}")

# Example: Train for a few episodes
train_dqn(episodes=5)

env.close()
