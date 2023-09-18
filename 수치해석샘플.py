import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 데이터 생성 (실제로는 전기 필요량 데이터를 불러와야 함)
n_data_points = 1000
x_data = np.linspace(0, 10, n_data_points).reshape(-1, 1)
y_data = 2 * np.sin(x_data) + 3 * np.cos(2 * x_data) + np.random.normal(0, 0.5, size=(n_data_points, 1))

# Numpy array를 PyTorch Tensor로 변환
x_tensor = torch.FloatTensor(x_data)
y_tensor = torch.FloatTensor(y_data)

# 데이터셋과 데이터로더 준비
dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# 모델 정의
class ElectricDemandModel(nn.Module):
    def __init__(self):
        super(ElectricDemandModel, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 모델, 손실 함수, 옵티마이저 초기화
model = ElectricDemandModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 훈련 루프
n_epochs = 1000
for epoch in range(n_epochs):
    for batch_x, batch_y in data_loader:
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

# 예측
with torch.no_grad():
    test_x = torch.FloatTensor(np.linspace(0, 10, 100).reshape(-1, 1))
    predicted = model(test_x)

# 결과 그리기
plt.scatter(x_data, y_data, label='True')
plt.plot(np.linspace(0, 10, 100), predicted.numpy(), label='Predicted', color='red')
plt.legend()
plt.show()