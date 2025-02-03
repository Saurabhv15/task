# Predicting Stock Levels Using Sales and Sensor Data

## Project Overview
This project aims to predict **stock levels** of products on an **hourly basis** by integrating sales data and sensor data (stock levels and storage temperature). Using a **Random Forest Regressor**, we train a machine learning model to identify significant features and provide actionable insights to the business for better procurement strategies.

## Features
1. **Data Integration**: Merge sales, stock, and sensor data into a single dataset.
2. **Feature Engineering**: Extract meaningful features like day, hour, and categorical encodings for modeling.
3. **Model Training**: Use a **Random Forest Regressor** with **K-Fold Cross Validation** to ensure robust predictions.
4. **Feature Importance Analysis**: Identify key factors influencing stock predictions.

---

## Prerequisites
To run this project, ensure you have the following:
- **Python 3.x**
- Libraries: `pandas`, `numpy`, `datetime`, `sklearn`, `matplotlib`
- Dataset files:
  - `sample_sales_data.csv`
  - `sensor_stock_levels.csv`
  - `sensor_storage_temperature.csv`

---

## Steps to Execute

### 1. Data Loading and Cleaning
Load the datasets and preprocess them:
- Remove unnecessary columns.
- Convert `timestamp` columns into a datetime format.
```python
sales_df = pd.read_csv('sample_sales_data.csv')
stock_df = pd.read_csv('sensor_stock_levels.csv')
temp_df = pd.read_csv('sensor_storage_temperature.csv')

sales_df['timestamp'] = pd.to_datetime(sales_df['timestamp'])
stock_df['timestamp'] = pd.to_datetime(stock_df['timestamp'])
temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'])
```

### 2. Aggregating and Merging Data
Transform timestamps to hourly format and aggregate datasets:
```python
def convert_timestamp_to_hourly(data, column):
    data[column] = data[column].dt.floor('H')
    return data

sales_df = convert_timestamp_to_hourly(sales_df, 'timestamp')
stock_df = convert_timestamp_to_hourly(stock_df, 'timestamp')
temp_df = convert_timestamp_to_hourly(temp_df, 'timestamp')

sales_agg = sales_df.groupby(['timestamp', 'product_id']).agg({'quantity': 'sum'}).reset_index()
stock_agg = stock_df.groupby(['timestamp', 'product_id']).agg({'estimated_stock_pct': 'mean'}).reset_index()
temp_agg = temp_df.groupby(['timestamp']).agg({'temperature': 'mean'}).reset_index()

merged_df = stock_agg.merge(sales_agg, on=['timestamp', 'product_id'], how='left')
merged_df = merged_df.merge(temp_agg, on='timestamp', how='left')
merged_df['quantity'] = merged_df['quantity'].fillna(0)
```

### 3. Feature Engineering
Create new features and encode categorical variables:
```python
merged_df['timestamp_day_of_month'] = merged_df['timestamp'].dt.day
merged_df['timestamp_day_of_week'] = merged_df['timestamp'].dt.dayofweek
merged_df['timestamp_hour'] = merged_df['timestamp'].dt.hour
merged_df = pd.get_dummies(merged_df, columns=['category'], drop_first=True)
merged_df.drop(columns=['timestamp', 'product_id'], inplace=True)
```

### 4. Model Training
Train a **Random Forest Regressor** to predict `estimated_stock_pct`:
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

X = merged_df.drop(columns=['estimated_stock_pct'])
y = merged_df['estimated_stock_pct']
accuracy = []
K = 10
split = 0.75

for fold in range(K):
    model = RandomForestRegressor()
    scaler = StandardScaler()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split, random_state=42)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    trained_model = model.fit(X_train, y_train)
    y_pred = trained_model.predict(X_test)

    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    accuracy.append(mae)
    print(f"Fold {fold + 1}: MAE = {mae:.3f}")

print(f"Average MAE: {(sum(accuracy) / len(accuracy)):.2f}")
```

### 5. Feature Importance
Visualize feature importance to understand key drivers:
```python
import matplotlib.pyplot as plt
import numpy as np

features = X.columns
importances = model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='y', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
```

---

## Key Findings
1. **Important Features**:
   - `unit_price` and `temperature` significantly impact stock levels.
   - `hour of the day` is also a crucial predictor.
2. **Model Performance**:
   - The Mean Absolute Error (MAE) is consistent across K-Folds but could improve with more engineered data or additional datasets.
3. **Business Insights**:
   - Recommend collecting more granular data or additional features to reduce the MAE and improve prediction accuracy.

---

## Next Steps
- Explore additional sensor data or external factors influencing stock levels.
- Optimize the machine learning pipeline with advanced models like XGBoost or neural networks.
- Collaborate with stakeholders to ensure meaningful implementation in supply chain management.

---

## Acknowledgments
This project was part of the **Cognizant Forage Virtual Internship**, designed to provide insights into real-world data analysis and machine learning workflows.

--- 

## Contact
For questions or collaborations, please contact **Saurabh Verma**.
