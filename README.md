# Numerical-analysis

### 에어컨 목표온도에 필요한 전력량 수치해석 샘플링

```
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
```
- 모듈 로드

```
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

```
- 데이터 예시(현재 온도, 목표 온도, 습도)

```
X_train_tensor = torch.from_numpy(X_train)
y_train_tensor = torch.from_numpy(y_train)
```
- numpy array를 PyTorch tensor로 변환

```
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
```
- 간단한 모델정의

```
net = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)
```
- 모델 객체 생성 및 손실 함수와 옵티마이저 정의

```
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
```
- 학습루프

```
test_data = torch.Tensor([[33, 23, 40]])  # 현재 온도, 목표 온도, 습도
predicted_power = net(test_data).item()
print(f'Predicted power requirement for the test data: {predicted_power:.2f} watts')
```
- 예측 테스트


### 실제 상황에 적용하려면
1. 더 많은 데이터가 필요
2. 특성 엔지니어링을 통해 모델의 성능을 높일 수 있음


### 특성 엔지니어링
특성 엔지니어링 : 머신 런이 알고리즘이나 모델이 문제를 더 해결할 수 있도록 원본 데이터를 변환하거나 개선하는 과정
1. 특성 스케일링(Feature Scaling) : 데이터의 범위를 일정하게 맞춰주는 것, 정규화(normalization)와 표준화(standardization)가 있음
2. 카테고리 데이터 수치화(Encoding Categorical Data) : 문자열이나 레이블 형태의 카테고리 데이터를 모델이 이해할 수 있는 숫자로 변환, 원-핫 인코딩이 있음
3. 결측치 처리(Handling Missing Values) : 누락된 값을 대체하거나 제거
4. 다항 특성과 교차 특성 생성(Polynomial and Interaction Features) : 기존 특성을 조합하여 새로운 특성을 만듬, '길이'와 '너비'라는 두 특성일 곱하면 '면적'이라는 새로운 특성이 생김
5. 비선형 변환(Non-linear Transformations) : 로그, 제곱근, 제곱 등의 수학적 변환을 적용
6. 특성 분해(Feature Decomposition) : 특성의 차원을 줄이거나 변환하여 새로운 특성을 생성, 주성분 분석(PCA)가 있음
7. 시간적 특성 추가(Temporal Features) : 날짜와 시간에서 여러 유용한 특성을 추출할 수 있음, 요일, 계절, 휴일, 여부 등을 새로운 특성으로 만들 수 있음
8. 도메인 지식 활용(Domain-specific Features) : 문제에 특화된 도메인 지식을 활용하여 새로운 특성을 생성

특성 엔지니어링은 모델의 성능을 크게 향상시킬 수 있지만, 도메인 지식과 데이터에 대한 깊은 이해가 필요하며 때로는 시간이 많이 소요될 수 있습니다.
따라서 이 과정은 주의 깊게 수행되어야 합니다.
