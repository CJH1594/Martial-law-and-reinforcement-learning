import numpy as np

class MarketEnv:
    def __init__(self, initial_state, fundamental_value=1.6666, num_steps=50, k1=0.1, k2=0.05, k3=0.02, k4=0.01, k5=0.01, k6=0.2, k7=0.1):
        self.initial_state = initial_state.copy()
        self.state = initial_state.copy()
        self.fundamental_value = fundamental_value
        self.num_steps = num_steps
        self.k1 = k1  # 주가 변화 민감도
        self.k2 = k2  # 환율 변화 민감도 (외국인 순매수)
        self.k3 = k3  # 금리 변화 민감도
        self.k4 = k4  # 금리가 환율에 미치는 영향
        self.k5 = k5  # 연기금의 거래 threashold
        self.k6 = k6  # 주가 방어 매수 보상 민감도
        self.k7 = k7  # 주가 조정 매도 보상 민감도
        self.current_step = 0

    def reset(self):
        self.state = self.initial_state.copy()
        self.current_step = 0
        return self._get_state()

    def _get_state(self):
        return {
            "stock_price": self.state["stock_price"],
            "exchange_rate": self.state["exchange_rate"],
            "bond_yield": self.state["bond_yield"],
            "fundamental_value": self.fundamental_value,
        }

    def step(self, actions):
        foreign_action, fund_action = actions
        total_net_buy = foreign_action + fund_action

        # 시장 변수 변화
        stock_price_change = self.k1 * total_net_buy
        exchange_rate_change = self.k2 * foreign_action + self.k4 * self.state["bond_yield"]
        bond_yield_change = self.k3 * total_net_buy

        self.state["stock_price"] += stock_price_change
        self.state["exchange_rate"] += exchange_rate_change
        self.state["bond_yield"] = max(0, self.state["bond_yield"] + bond_yield_change)  # 음수 금리 방지

        self.current_step += 1
        done = (self.current_step >= self.num_steps)

        # 외국인 보상
        foreign_reward = -abs(self.state["stock_price"] - self.fundamental_value * self.state["exchange_rate"])

        # 연기금 보상: k5보다 낮으면 음수 보상, 높으면 보상 0
        fund_reward = -self.k6 * max(0, self.k5 - self.state["stock_price"])

        rewards = [foreign_reward, fund_reward]

        return self._get_state(), rewards, done


# Usage Example
if __name__ == "__main__":
    initial_state = {
        "stock_price": 2400,
        "exchange_rate": 1400,
        "bond_yield": 1.5,
    }

    env = MarketEnv(initial_state=initial_state, fundamental_value=1.6666, num_steps=50)

    state = env.reset()
    for step in range(10):
        actions = [0.5, -0.3]  # 외국인 매수, 연기금 매도 (예시)
        next_state, rewards, done = env.step(actions)
        print(f"Step {step + 1}: State={next_state}, Rewards={rewards}")
        if done:
            break
