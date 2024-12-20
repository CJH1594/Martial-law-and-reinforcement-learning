import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # [-1, 1] 범위의 액션 출력
        )

    def forward(self, state):
        return self.fc(state)

class DQNAgent:
    def __init__(self, state_dim, action_dim, buffer_size=10000, batch_size=32, gamma=0.99, tau=0.005, lr_actor=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.actor = ActorNetwork(state_dim, action_dim)
        self.actor_target = ActorNetwork(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.replay_buffer = deque(maxlen=buffer_size)

    def get_action(self, state, noise=0.1):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state_tensor).detach().numpy()[0]
        action += noise * np.random.randn(self.action_dim)  # 노이즈 추가
        return np.clip(action, -1.0, 1.0)

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update_policy(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states = torch.FloatTensor([x[0] for x in batch])
        actions = torch.FloatTensor([x[1] for x in batch])
        rewards = torch.FloatTensor([x[2] for x in batch]).unsqueeze(1)
        next_states = torch.FloatTensor([x[3] for x in batch])
        dones = torch.FloatTensor([x[4] for x in batch]).unsqueeze(1)

        target_actions = self.actor_target(next_states)
        q_targets = rewards + self.gamma * target_actions * (1 - dones)
        critic_loss = nn.MSELoss()(self.actor(states), q_targets)

        self.actor_optimizer.zero_grad()
        critic_loss.backward()
        self.actor_optimizer.step()

    def state_dict(self):
        """에이전트의 상태(모델 파라미터)를 반환"""
        return {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """저장된 상태를 로드"""
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_target.load_state_dict(state_dict["actor_target"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])