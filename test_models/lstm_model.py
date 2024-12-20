import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

# 데이터 로드
data = pd.read_csv("./data/processed_data.csv")

# 데이터 전처리
df = pd.DataFrame(data)
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

# 입력과 출력 데이터 분리
features = ["stock_foreign_net_buy", "stock_pension_net_buy", "stock_individual_net_buy", "bond_close_price", "bond_volume", "exchange_rate"]
target = "stock_close_price"

# 타깃 변수의 최소값과 최대값 계산
target_min = df[target].min()
target_max = df[target].max()

# 전체 데이터 정규화
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[features + [target]])

# 시퀀스 생성 함수
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length, :-1]
        label = data[i + seq_length, -1]
        sequences.append(seq)
        targets.append(label)
    return np.array(sequences), np.array(targets)

seq_length = 3
sequences, targets = create_sequences(data_scaled, seq_length)

# 데이터셋 분리
X_train, X_test, y_train, y_test = train_test_split(sequences, targets, test_size=0.2, random_state=42)

# Tensor 변환
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 예측 결과 역정규화 함수
def inverse_transform(predictions, target_min, target_max):
    return predictions * (target_max - target_min) + target_min

# LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

input_size = len(features)
hidden_size = 64
num_layers = 2
output_size = 1

model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# 손실 함수와 옵티마이저
epochs = 100
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 모델 학습
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output.squeeze(), y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# 테스트 데이터로 평가
model.eval()
with torch.no_grad():
    predictions = model(X_test).squeeze()
    test_loss = criterion(predictions, y_test)
    print(f"Test Loss: {test_loss.item():.4f}")

# 결과 출력
model_predictions_denormalized = inverse_transform(predictions, target_min, target_max)
model_true_denormalized = inverse_transform(y_test, target_min, target_max)
print("Predictions:", model_predictions_denormalized.numpy())
print("True Values:", model_true_denormalized.numpy())
