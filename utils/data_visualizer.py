import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def plot_market_data_and_trading_volumes(simulated_data, real_data):
    # 시뮬레이션 데이터 준비
    stock_prices_sim = simulated_data[:, 0]
    exchange_rates_sim = simulated_data[:, 1]
    bond_yields_sim = simulated_data[:, 2]
    foreign_volumes_sim = simulated_data[:, 3]
    fund_volumes_sim = simulated_data[:, 4]

    # 실제 데이터 준비
    stock_prices_real = real_data["stock_price"].values
    exchange_rates_real = real_data["exchange_rate"].values
    bond_yields_real = real_data["bond_yield"].values

    time_steps = range(len(stock_prices_sim))

    # 그래프 크기 설정 (80% 크기로 축소)
    plt.figure(figsize=(11.2, 8))  # 원래 크기 (14, 10)의 80%

    # Stock Price
    plt.subplot(2, 2, 1)
    plt.plot(time_steps, stock_prices_sim, label="Simulated Stock Price", color="blue")
    plt.plot(time_steps, stock_prices_real, label="Real Stock Price", color="red", linestyle="dashed")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))  # Float 형태로 표시

    # Exchange Rate
    plt.subplot(2, 2, 2)
    plt.plot(time_steps, exchange_rates_sim, label="Simulated Exchange Rate", color="orange")
    plt.plot(time_steps, exchange_rates_real, label="Real Exchange Rate", color="red", linestyle="dashed")
    plt.xlabel("Time Step")
    plt.ylabel("Rate")
    plt.legend()
    plt.grid()
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))  # Float 형태로 표시

    # Bond Yield
    plt.subplot(2, 2, 3)
    plt.plot(time_steps, bond_yields_sim, label="Simulated Bond Yield", color="green")
    plt.plot(time_steps, bond_yields_real, label="Real Bond Yield", color="red", linestyle="dashed")
    plt.xlabel("Time Step")
    plt.ylabel("Yield")
    plt.legend()
    plt.grid()
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))  # Float 형태로 표시

    # Trading Volumes by Agent
    plt.subplot(2, 2, 4)
    plt.plot(time_steps, foreign_volumes_sim, label="Simulated Foreign Volume", color="blue")
    plt.plot(time_steps, fund_volumes_sim, label="Simulated Fund Volume", color="green")
    plt.xlabel("Time Step")
    plt.ylabel("Volume")
    plt.legend()
    plt.grid()
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))  # Float 형태로 표시

    # 레이아웃 조정 및 표시
    plt.tight_layout()
    plt.show()

def plot_simulation(simulated_data):
    """
    시뮬레이션 데이터를 기반으로 시장 변수와 거래량을 시각화한다.

    Args:
        simulated_data (np.ndarray): 시뮬레이션 데이터 배열 (shape: [time_steps, 5])
    """
    # 시뮬레이션 데이터 준비
    stock_prices_sim = simulated_data[:, 0]
    exchange_rates_sim = simulated_data[:, 1]
    bond_yields_sim = simulated_data[:, 2]
    foreign_volumes_sim = simulated_data[:, 3]
    fund_volumes_sim = simulated_data[:, 4]

    time_steps = range(len(stock_prices_sim))

    # 그래프 크기 설정 (80% 크기로 축소)
    plt.figure(figsize=(11.2, 8))  # 원래 크기 (14, 10)의 80%

    # Stock Price
    plt.subplot(2, 2, 1)
    plt.plot(time_steps, stock_prices_sim, label="Simulated Stock Price", color="blue")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))  # Float 형태로 표시

    # Exchange Rate
    plt.subplot(2, 2, 2)
    plt.plot(time_steps, exchange_rates_sim, label="Simulated Exchange Rate", color="orange")
    plt.xlabel("Time Step")
    plt.ylabel("Rate")
    plt.legend()
    plt.grid()
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))  # Float 형태로 표시

    # Bond Yield
    plt.subplot(2, 2, 3)
    plt.plot(time_steps, bond_yields_sim, label="Simulated Bond Yield", color="green")
    plt.xlabel("Time Step")
    plt.ylabel("Yield")
    plt.legend()
    plt.grid()
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))  # Float 형태로 표시

    # Trading Volumes by Agent
    plt.subplot(2, 2, 4)
    plt.plot(time_steps, foreign_volumes_sim, label="Simulated Foreign Volume", color="blue")
    plt.plot(time_steps, fund_volumes_sim, label="Simulated Fund Volume", color="green")
    plt.xlabel("Time Step")
    plt.ylabel("Volume")
    plt.legend()
    plt.grid()
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))  # Float 형태로 표시

    # 레이아웃 조정 및 표시
    plt.tight_layout()
    plt.show()
