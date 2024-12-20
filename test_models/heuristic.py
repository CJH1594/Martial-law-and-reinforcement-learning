# pseudo code
import torch
import torch.nn as nn
import numpy as np

# Agent 예시 클래스 (초기엔 규칙 기반)
class BaseAgent:
    def __init__(self, name, init_portfolio=100000, init_position_ratio=0.5):
        self.name = name
        self.portfolio = init_portfolio  # 전체 자산 가치
        self.position_ratio = init_position_ratio  # 자산 중 주식 등 위험자산 비중
        self.recent_action = 0.0
        
    def get_state(self, market_state):
        # market_state: { 'stock_price': float, 'bond_price': float, 'exchange_rate': float, ... }
        # state: 보유 자산 비중, 최근동향, 시장변동
        state = {
            'portfolio': self.portfolio,
            'position_ratio': self.position_ratio,
            'stock_price': market_state['stock_price'],
            'bond_price': market_state['bond_price'],
            'exchange_rate': market_state['exchange_rate']
        }
        return state

    def decide_action(self, state):
        # 단순 규칙 기반 예: 주가 상승 추세면 조금 더 매수, 아니면 매도
        # 실제로는 지난 step 대비 가격 변화를 판단하여 결정
        # e.g., 현재 step 주가 > 이전 step 주가면 +0.01 매수, 아니면 -0.01 매도
        # position_ratio를 0 ~ 1 사이에서 변경
        if state['stock_price'] > 1400:  # 예시 기준
            action = 0.01  # 매수 비중 증가
        else:
            action = -0.01  # 매도 비중 증가
        return action

    def step(self, market_state):
        state = self.get_state(market_state)
        action = self.decide_action(state)

        # 행동 반영: 포지션 비중 업데이트
        new_ratio = self.position_ratio + action
        # 비중은 0~1 사이로 클리핑
        new_ratio = max(0.0, min(new_ratio, 1.0))

        # 포지션 조정에 따른 자산 가치 변화 가정 (단순화)
        # 예: position_ratio 만큼 주식 보유, 나머지 현금 or 채권으로 가정
        # 주가 변동으로 포트폴리오 가치 변화를 반영
        old_portfolio = self.portfolio
        stock_value = new_ratio * self.portfolio
        bond_cash_value = (1 - new_ratio) * self.portfolio

        # 주가, 채권가치 변화를 간단히 반영 (t+1에 계산)
        # 여기서는 일단 행동만 정하고, 실제 시장 업데이트 후 포트폴리오 재평가

        self.position_ratio = new_ratio
        self.recent_action = action

        # 보상 계산은 환경 업데이트 후에 진행
        return action


# 간단한 환경 클래스
class MarketEnv:
    def __init__(self, market_data):
        # market_data: time series of {date, stock_price, bond_price, exchange_rate, ...}
        self.data = market_data
        self.current_step = 0
        self.agents = []
        
    def add_agent(self, agent):
        self.agents.append(agent)

    def reset(self):
        self.current_step = 0
        # 필요하다면 에이전트 상태 초기화
        # 예를 들어 포지션, 포트폴리오 가치 리셋

    def get_market_state(self):
        # 현재 step의 시장 상태
        row = self.data[self.current_step]
        market_state = {
            'stock_price': row['stock_price'],
            'bond_price': row['bond_close_price'],
            'exchange_rate': row['exchange_rate']
        }
        return market_state

    def step(self):
        # 각 에이전트 행동
        market_state = self.get_market_state()
        actions = {}
        for agent in self.agents:
            actions[agent.name] = agent.step(market_state)

        # 에이전트들의 행동을 시장에 반영
        # 간단한 규칙: 모든 에이전트가 매수 우위 -> 주가 약간 상승, 매도우위 -> 하락
        net_action = sum(actions.values())
        # 예: net_action > 0이면 매수세, 주가 1% 상승 / net_action < 0이면 매도세, 주가 1% 하락 (초간단)
        # 실제로는 더 정교한 메커니즘 필요
        if net_action > 0:
            self.data[self.current_step+1]['stock_price'] *= (1 + 0.01 * net_action)
        else:
            self.data[self.current_step+1]['stock_price'] *= (1 + 0.01 * net_action)
        
        # 한 스텝 진행
        self.current_step += 1

        # 보상 계산: 각 에이전트의 포트폴리오가 다음 시점에서 어떻게 변화했는지 반영
        # 현재시점 포트폴리오 재평가
        # 매우 단순화: agent의 포지션 비율 * 새로운 주가로 가치 변화 계산
        next_state = self.get_market_state()
        for agent in self.agents:
            agent.portfolio = agent.position_ratio * agent.portfolio * (next_state['stock_price'] / market_state['stock_price']) \
                              + (1 - agent.position_ratio) * agent.portfolio
            # 여기서는 포트폴리오 전체를 주식과 현금(또는 안전자산)으로만 단순 가정
            # 보상 = 새로운 포트폴리오 가치 - 이전 포트폴리오 가치
            reward = agent.portfolio - (agent.position_ratio * agent.portfolio / (next_state['stock_price']/market_state['stock_price'])
                                        + (1 - agent.position_ratio)*agent.portfolio)
            # 실제로는 이전 step에서의 포트폴리오 가치 기록이 필요하거나, step 진행 전 값 저장 필요

        done = (self.current_step >= len(self.data)-1)
        return done


# 간단한 실행 예시
if __name__ == "__main__":
    # 가상 market_data 생성 (실제 데이터 사용 가능)
    market_data = []
    for i in range(20):
        market_data.append({
            'stock_price': 1400 + i*2,  # 단순 증가 가정
            'bond_close_price': 100.0,
            'exchange_rate': 1300.0
        })

    env = MarketEnv(market_data)
    agent1 = BaseAgent(name='fund')
    agent2 = BaseAgent(name='foreigner')
    agent3 = BaseAgent(name='retail')

    env.add_agent(agent1)
    env.add_agent(agent2)
    env.add_agent(agent3)

    env.reset()
    done = False
    while not done:
        done = env.step()
