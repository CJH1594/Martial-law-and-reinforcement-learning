import torch
import numpy as np
import pandas as pd
import json
import optuna
from sklearn.preprocessing import MinMaxScaler
from models.market_env import MarketEnv
from models.dqn_agent import DQNAgent
from utils.data_visualizer import plot_market_data_and_trading_volumes

# 실측 데이터 로드 및 정렬
real_data = pd.read_csv("data/processed_data.csv")
real_data = real_data.sort_values(by="date").reset_index(drop=True)

# MinMaxScaler로 데이터 정규화
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(real_data[["stock_price", "exchange_rate", "bond_yield"]])
real_data[["stock_price", "exchange_rate", "bond_yield"]] = scaled_data

# 실험 설정
num_episodes = 100  # 각 파라미터 조합별 학습 에피소드 수
num_steps = len(real_data)  # 타임스텝은 실제 데이터 길이에 맞춤

# 최적화를 위한 목적 함수 정의
def objective(trial):
    # 파라미터 샘플링
    k1 = trial.suggest_float("k1", 0.01, 1.0)
    k2 = trial.suggest_float("k2", 0.01, 1.0)
    k3 = trial.suggest_float("k3", 0.01, 1.0)
    k4 = trial.suggest_float("k4", 0.01, 1.0)
    k5 = trial.suggest_float("k5", 0.01, 1.0)
    k6 = trial.suggest_float("k6", 0.01, 1.0)
    k7 = trial.suggest_float("k7", 0.01, 1.0)

    # 환경 생성
    env = MarketEnv(
        initial_state={
            "stock_price": real_data.iloc[0]["stock_price"],
            "exchange_rate": real_data.iloc[0]["exchange_rate"],
            "bond_yield": real_data.iloc[0]["bond_yield"],
        },
        fundamental_value=real_data.iloc[-1]["stock_price"] / real_data.iloc[-1]["exchange_rate"],
        num_steps=num_steps,
        k1=k1, k2=k2, k3=k3, k4=k4, k5=k5, k6=k6, k7=k7
    )

    # 에이전트 초기화
    foreigner = DQNAgent(state_dim=4, action_dim=1)
    fund = DQNAgent(state_dim=4, action_dim=1)

    # 학습
    for episode in range(num_episodes):
        state = env.reset()
        for step in range(num_steps):
            state_vec = [
                state["stock_price"],
                state["exchange_rate"],
                state["bond_yield"],
                state["fundamental_value"],
            ]
            foreign_action = foreigner.get_action(state_vec)
            fund_action = fund.get_action(state_vec)

            next_state, rewards, done = env.step([foreign_action[0], fund_action[0]])

            # 경험 저장 및 정책 업데이트
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

    # 평가
    simulated_data = np.zeros((num_steps, 3))
    state = env.reset()
    for step in range(num_steps):
        state_vec = [
            state["stock_price"],
            state["exchange_rate"],
            state["bond_yield"],
            state["fundamental_value"],
        ]
        foreign_action = foreigner.get_action(state_vec, noise=0.0)
        fund_action = fund.get_action(state_vec, noise=0.0)

        next_state, _, _ = env.step([foreign_action[0], fund_action[0]])

        simulated_data[step] = [
            state["stock_price"],
            state["exchange_rate"],
            state["bond_yield"],
        ]
        state = next_state

    # MSE 계산
    mse_stock_price = np.mean((simulated_data[:, 0] - real_data["stock_price"].values) ** 2)
    mse_exchange_rate = np.mean((simulated_data[:, 1] - real_data["exchange_rate"].values) ** 2)
    mse_bond_yield = np.mean((simulated_data[:, 2] - real_data["bond_yield"].values) ** 2)

    total_mse = mse_stock_price + mse_exchange_rate + mse_bond_yield
    return total_mse

# Optuna로 최적화 수행
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# 최적 파라미터 출력
best_params = study.best_params
best_mse = study.best_value

print(f"Best Params: {best_params}")
print(f"Best MSE: {best_mse:.4f}")

# 최적 파라미터로 시뮬레이션
env = MarketEnv(
    initial_state={
        "stock_price": real_data.iloc[0]["stock_price"],
        "exchange_rate": real_data.iloc[0]["exchange_rate"],
        "bond_yield": real_data.iloc[0]["bond_yield"],
    },
    fundamental_value=real_data.iloc[-1]["stock_price"] / real_data.iloc[-1]["exchange_rate"],
    num_steps=num_steps,
    **best_params
)

# 에이전트 초기화
foreigner = DQNAgent(state_dim=4, action_dim=1)  # 여기서 foreigner 다시 정의
fund = DQNAgent(state_dim=4, action_dim=1)       # fund 역시 다시 정의

# 시뮬레이션 데이터 저장 및 시각화
market_data_history = np.zeros((num_steps, 5))
state = env.reset()
for step in range(num_steps):
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

plot_market_data_and_trading_volumes(market_data_history, real_data)

# 외국인과 연기금 모델 저장
torch.save(foreigner.state_dict(), "best_foreigner_model.pth")
torch.save(fund.state_dict(), "best_fund_model.pth")

# 최적 파라미터 저장
with open("best_params.json", "w") as f:
    json.dump(best_params, f)

print("최적의 모델(Foreigner, Fund)과 파라미터가 저장되었습니다.")
