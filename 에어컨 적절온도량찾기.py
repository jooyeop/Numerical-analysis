import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 데이터 예시 (현재 온도, 목표 온도, 습도)
X_train = np.array([[25, 20, 30],
                    [30, 20, 40],
                    [35, 25, 35],
                    [40, 30, 50],
                    [20, 18, 20]], dtype=np.float32)

# 목표 전력량 예시
y_train = np.array([[100],
                    [150],
                    [200],
                    [220],
                    [90]], dtype=np.float32)

# numpy array를 PyTorch tensor로 변환
X_train_tensor = torch.from_numpy(X_train)
y_train_tensor = torch.from_numpy(y_train)

# 모델 정의
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 모델 객체 생성
net = Net()

# 손실 함수와 옵티마이저 정의
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

# 학습 루프
for epoch in range(1000):
    # Forward pass
    outputs = net(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Zero gradients, backward pass, optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 로그 출력
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')

# 예측 테스트
test_data = torch.Tensor([[33, 23, 40]])  # 현재 온도, 목표 온도, 습도
predicted_power = net(test_data).item()
print(f'Predicted power requirement for the test data: {predicted_power:.2f} watts')
