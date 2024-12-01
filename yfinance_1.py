import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. 주가 데이터 가져오기
ticker = 'MSFT'  # 애플 주식
data = yf.download(ticker, start='2020-01-01', end='2023-01-01')

# 2. 데이터 전처리
data['Date'] = data.index
data['Days'] = (data['Date'] - data['Date'].min()).dt.days  # 날짜를 숫자로 변환
X = data[['Days']]  # 입력 변수
y = data['Close']  # 목표 변수 (종가)

# 3. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 모델 훈련
model = LinearRegression()
model.fit(X_train, y_train)

# 5. 예측
predictions = model.predict(X_test)

# 6. 결과 시각화
plt.figure(figsize=(10, 5))
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.scatter(X_test, predictions, color='red', label='Predicted Prices')
#plt.title('Stock Price Prediction')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Days since start')
plt.ylabel('Stock Price')
plt.legend()
plt.show()