import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class BaseAgent:
    """기본 에이전트 클래스
    모든 에이전트는 이 클래스를 상속받아 구현
    """

    def __init__(self, name, action_space):
        """
        Args:
            name (str): 에이전트 이름
            action_space (int): 가능한 이산적 행동의 수 (예: 3 -> 매도, 홀드, 매수)
        """
        self.name = name
        self.action_space = action_space

    def get_action(self, state):
        """현재 상태에서 행동을 선택하는 메서드 (여기서는 추상 메서드로 정의)"""
        raise NotImplementedError

    def store_experience(self, state, action, reward, next_state, done):
        """경험(transition) 저장 메서드"""
        pass

    def update_policy(self):
        """정책 업데이트 메서드(학습 진행)"""
        pass


class QNetwork(nn.Module):
    """단순한 Q값 추정용 신경망: 상태 -> 각 행동에 대한 Q값"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.fc(x)


class DQNAgent(BaseAgent):
    """DQN 기반 에이전트 예시"""
    def __init__(self, name, state_dim, action_space, hidden_dim=64, lr=0.001, gamma=0.99, epsilon=0.1, buffer_size=10000, batch_size=32):
        super().__init__(name, action_space)
        self.state_dim = state_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size

        # Q-Network 초기화
        self.q_network = QNetwork(state_dim, action_space, hidden_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # 경험 리플레이 버퍼
        self.replay_buffer = deque(maxlen=buffer_size)

    def get_action(self, state):
        """epsilon-greedy 정책으로 행동 선택"""
        if np.random.rand() < self.epsilon:
            # 무작위 행동
            return np.random.randint(0, self.action_space)
        else:
            # Q값이 최대인 행동 선택
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                action = torch.argmax(q_values).item()
                return action

    def store_experience(self, state, action, reward, next_state, done):
        """환경 상호작용 결과를 리플레이 버퍼에 저장"""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update_policy(self):
        """리플레이 버퍼에서 샘플을 뽑아 Q-network 업데이트"""
        if len(self.replay_buffer) < self.batch_size:
            return  # 샘플이 충분하지 않을 때는 학습 패스

        batch = random.sample(self.replay_buffer, self.batch_size)
        states = torch.FloatTensor([x[0] for x in batch])
        actions = torch.LongTensor([x[1] for x in batch]).unsqueeze(-1)
        rewards = torch.FloatTensor([x[2] for x in batch]).unsqueeze(-1)
        next_states = torch.FloatTensor([x[3] for x in batch])
        dones = torch.FloatTensor([x[4] for x in batch]).unsqueeze(-1)

        # 현재 Q값
        q_values = self.q_network(states).gather(1, actions)

        # 다음 상태에서의 최대 Q값
        with torch.no_grad():
            max_next_q_values = self.q_network(next_states).max(dim=1, keepdim=True)[0]
            target = rewards + (1 - dones) * self.gamma * max_next_q_values

        loss = self.criterion(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
