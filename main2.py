import numpy as np
import pandas as pd
import json
import torch
from sklearn.preprocessing import MinMaxScaler
from models.market_env import MarketEnv
from models.dqn_agent import DQNAgent
from utils.data_visualizer import plot_simulation

# 실측 데이터 로드 및 정렬
real_data = pd.read_csv("data/processed_data.csv")
real_data = real_data.sort_values(by="date").reset_index(drop=True)

# 데이터 정규화
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(real_data[["stock_price", "exchange_rate", "bond_yield"]])
real_data[["stock_price", "exchange_rate", "bond_yield"]] = scaled_data

# 기존 모델 및 파라미터 로드
with open("best_params.json", "r") as f:
    best_params = json.load(f)

# k5 값을 0으로 설정
new_k_values = {
    "k1": best_params["k1"],
    "k2": best_params["k2"],
    "k3": best_params["k3"],
    "k4": best_params["k4"],
    "k5": 0.0,  # k5를 0으로 설정
    "k6": best_params["k6"],
    "k7": best_params["k7"],
}

# 30일 시뮬레이션
num_steps = 30 

# 새로운 환경 생성
env = MarketEnv(
    initial_state={
        "stock_price": real_data.iloc[-1]["stock_price"],  # 마지막 시점 데이터를 초기 상태로 설정
        "exchange_rate": real_data.iloc[-1]["exchange_rate"],
        "bond_yield": real_data.iloc[-1]["bond_yield"],
    },
    fundamental_value=(real_data.iloc[-1]["stock_price"] * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]) /
                      (real_data.iloc[-1]["exchange_rate"] * (scaler.data_max_[1] - scaler.data_min_[1]) + scaler.data_min_[1]),
    num_steps=num_steps,  # 30일치 시뮬레이션
    **new_k_values
)

# 모델 초기화
foreigner = DQNAgent(state_dim=4, action_dim=1)
fund = DQNAgent(state_dim=4, action_dim=1)

# 학습
num_episodes = 100
for episode in range(num_episodes):
    state = env.reset()
    for step in range(len(real_data)):
        state_vec = [
            state["stock_price"],
            state["exchange_rate"],
            state["bond_yield"],
            state["fundamental_value"],
        ]
        foreign_action = foreigner.get_action(state_vec)
        fund_action = fund.get_action(state_vec)

        next_state, rewards, done = env.step([foreign_action[0], fund_action[0]])

        next_state_vec = [
            next_state["stock_price"],
            next_state["exchange_rate"],
            next_state["bond_yield"],
            next_state["fundamental_value"],
        ]
        foreigner.store_experience(state_vec, foreign_action[0], rewards[0], next_state_vec, done)
        fund.store_experience(state_vec, fund_action[0], rewards[1], next_state_vec, done)

        foreigner.update_policy()
        fund.update_policy()

        state = next_state
        if done:
            break

# 시뮬레이션 및 데이터 저장
market_data_history = np.zeros((num_steps, 5))  # 30일치 데이터를 저장할 배열
state = env.reset()
for step in range(num_steps):  # 30번 반복
    state_vec = [
        state["stock_price"],
        state["exchange_rate"],
        state["bond_yield"],
        state["fundamental_value"],
    ]
    foreign_action = foreigner.get_action(state_vec, noise=0.0)
    fund_action = fund.get_action(state_vec, noise=0.0)

    next_state, _, _ = env.step([foreign_action[0], fund_action[0]])

    market_data_history[step] = [
        state["stock_price"],
        state["exchange_rate"],
        state["bond_yield"],
        foreign_action[0],
        fund_action[0],
    ]
    state = next_state


# 역정규화
market_data_history[:, 0] = market_data_history[:, 0] * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
market_data_history[:, 1] = market_data_history[:, 1] * (scaler.data_max_[1] - scaler.data_min_[1]) + scaler.data_min_[1]
market_data_history[:, 2] = market_data_history[:, 2] * (scaler.data_max_[2] - scaler.data_min_[2]) + scaler.data_min_[2]

# 시각화
plot_simulation(market_data_history)