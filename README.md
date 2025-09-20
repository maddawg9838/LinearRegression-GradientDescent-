# Linear Regression Project

## Overview
This project implements linear regression in two parts:

1. **Part 1:** Custom implementation of linear regression using gradient descent (no ML libraries)
  - Libraries Used:
    - numpy
    - pandas
    - matplotlib (pyplot)
    - seaborn
    -  sklearn.model_selection (train_test_split)
    -  sklearn.preprocessing (StandardScaler)
    -  sklearn.metrics (mean_squared_error, mean_absolute_error, r2_score, explained_variance_score)
2. **Part2:** Linear regression using scikit-learn's 'SGDRegressor'.
  - Libraries Used:
    - pandas
    - matplotlib (pyplot)
    -  sklearn.model_selection (train_test_split)
    -  sklearn.preprocessing (StandardScaler)
    -  sklearn.metrics (mean_squared_error, mean_absolute_error, r2_score, explained_variance_score)
    -  sklearn.linear_model (SGDRegressor)

Both parts use the **Daily Demand Forecasting Orders** datasets. 

Preprocessing includes:
1. Feature Scaling
2. Correlation Analysis
3. Feature Selection
4. Train/Test Split

---

## Requirements

### Python Version
- Python 3.9 or higher (recommended)

## How to run
### Create and activate a Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
```

### Install Required Libraries
```bash
pip install -r requirements.txt
```

### Part 1: Custom Gradient Descent
```bash
python part1.py
```

### Part 2: Scikit-learn SGDRegressor
```bash
python part2.py
```
