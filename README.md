# ☁️ 全球天氣數據分析專案
Global Weather Data Analysis Project

## 📌 專案資訊 | Project Information
- 作者 | Author：jeanbomb
- 更新 | Updated：2025-01-14 
- 資料集 | Dataset：Global Weather Repository (Kaggle)
- Kaggle Notebook：[Weather Analysis](https://www.kaggle.com/code/game1g/weather)

## 🎯 專案概述 | Project Overview
本專案利用機器學習技術對全球天氣數據進行分析，比較線性回歸和隨機森林兩種模型在天氣預測上的表現。透過數據分析探索全球天氣模式、空氣品質分布等環境特徵。

This project utilizes machine learning techniques to analyze global weather data, comparing Linear Regression and Random Forest models for weather prediction. The analysis explores global weather patterns and environmental characteristics including air quality distribution.

## 🛠️ 技術需求 | Technical Requirements
### Kaggle 環境所需套件 | Required Packages in Kaggle
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
```

## 📊 分析方法 | Analysis Methods
```python
# 載入數據
data = pd.read_csv("/kaggle/input/global-weather-repository/GlobalWeatherRepository.csv")

# 數據預處理
data = data[data['air_quality_PM2.5'] > 0]
data = data[data['air_quality_Carbon_Monoxide'] > 0]

# 特徵選擇
X = data[['latitude', 'longitude', 'humidity']]
y = data['temperature_celsius']

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 標準化數據
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 線性回歸模型
model_lr = LinearRegression()
model_lr.fit(X_train_scaled, y_train)
predictions_lr = model_lr.predict(X_test_scaled)
mse_lr = mean_squared_error(y_test, predictions_lr)
r2_lr = r2_score(y_test, predictions_lr)

# 隨機森林模型
model_rf = RandomForestRegressor()
model_rf.fit(X_train_scaled, y_train)
predictions_rf = model_rf.predict(X_test_scaled)
mse_rf = mean_squared_error(y_test, predictions_rf)
r2_rf = r2_score(y_test, predictions_rf)
```

### 1. 線性回歸分析 | Linear Regression Analysis
- 使用特徵：緯度、經度、濕度
- 目標變數：溫度（攝氏）
- 模型評估：
  * MSE: 60.89
  * R² Score: 0.228

### 2. 隨機森林分析 | Random Forest Analysis
- 使用相同特徵集
- 模型評估：
  * MSE: 11.17
  * R² Score: 0.858
- 展現出更好的預測性能

### 3. 數據預處理 | Data Preprocessing
- 清理異常值
  * 移除負值的 PM2.5 數據
  * 移除負值的一氧化碳數據
- 特徵標準化
- 訓練集/測試集切分 (80/20)

## 📈 分析結果 | Analysis Results
### 1. 溫度分布 | Temperature Distribution
最高平均溫度國家 (Top 3):
1. Saudi Arabien (45.0°C)
2. Marrocos (40.3°C)
3. Turkménistan (37.8°C)

### 2. 空氣品質 | Air Quality
PM2.5 污染最嚴重國家 (Top 3):
1. Chile (299.95)
2. China (143.55)
3. India (102.92)

### 3. 模型比較 | Model Comparison
- 隨機森林模型表現顯著優於線性回歸
- 說明天氣預測具有明顯的非線性特性

## 💻 使用方法 | Usage
### Kaggle Notebook 使用步驟
1. 點擊上方 Kaggle Notebook 連結
2. 點擊 "Copy and Edit" 創建自己的版本
3. 執行所有程式碼單元格
4. 可以根據需求修改參數進行新的分析

## 📊 數據說明 | Data Description
- 資料筆數：46,772 筆
- 特徵數量：41 個欄位
- 主要特徵：
  * 溫度（攝氏/華氏）
  * 風速與風向
  * 氣壓
  * 濕度
  * 空氣品質指標
