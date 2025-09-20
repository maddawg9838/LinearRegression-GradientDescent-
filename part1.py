import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

matplotlib.use("TkAgg")

# Linear Regression using Gradient Descent (from scratch)
class LinearRegressionGD:
    def __init__(self, lr=0.01, n_iter=1000, tol=1e-6, verbose=False):
        self.lr = lr
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self.w = None
        self.b = None
        self.history = []

    # Train the model using gradient descent
    def fit(self, X, y):
        X = self._check_input(X)
        y = self._check_input(y, vector=True)
        m, n = X.shape
        self.w = np.zeros((n, 1))
        self.b = 0

        for i in range(self.n_iter):
            y_pred = X.dot(self.w) + self.b
            error = y_pred - y
            self._update_weights(X, error, m)
            mse = (1 / m) * np.sum(error ** 2)
            self.history.append(mse)
            if i > 0 and abs(self.history[-2] - mse) < self.tol:
                if self.verbose:
                    print(f"Converged after {i} iterations")
                break
        if self.verbose:
            print(f"Final MSE: {self.history[-1]:.6f}")

    # Predict target values
    def predict(self, X):
        X = self._check_input(X)
        return X.dot(self.w) + self.b

    # Compute evaluation metrics
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return {
            "MSE": mean_squared_error(y, y_pred),
            "MAE": mean_absolute_error(y, y_pred),
            "R2": r2_score(y, y_pred),
            "EV": explained_variance_score(y, y_pred)
        }

    # Updating the Weights
    def _update_weights(self, X, error, m):
        self.w -= self.lr * (1 / m) * X.T.dot(error)
        self.b -= self.lr * (1 / m) * np.sum(error)

    # Ensure input is numpy array and properly shaped
    def _check_input(self, data, vector=False):
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            data = data.values
        if vector:
            data = data.reshape(-1, 1)
        return np.array(data)


class RegressionPlotter:
    def __init__(self):
        sns.set()

    def plot_mse_history(self, mse_history, title="MSE vs Iterations"):
        plt.figure(figsize=(8, 5))
        k = 50 # plot every 50th iteration
        plt.plot(range(1, len(mse_history) + 1, k), mse_history[::k], marker='o')
        plt.xlabel("Iteration")
        plt.ylabel("MSE")
        plt.title(title)
        plt.grid(True)
        plt.show()

    def plot_predicted_vs_actual(self, y_true, y_pred, dataset_name="Dataset"):
        y_true = np.ravel(y_true)
        y_pred = np.ravel(y_pred)

        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, alpha=0.7)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"Predicted vs Actual - {dataset_name}")
        plt.show()

    def plot_residuals(self, y_true, y_pred, dataset_name="Dataset"):
        residuals = y_true - y_pred
        plt.figure(figsize=(8, 5))
        plt.scatter(range(len(residuals)), residuals, alpha=0.7)
        plt.axhline(0, color='r', linestyle='--')
        plt.xlabel("Sample Index")
        plt.ylabel("Residual")
        plt.title(f"Residuals - {dataset_name}")
        plt.show()


def main():
    # 1. LOAD DATASET
    from ucimlrepo import fetch_ucirepo

    # fetch dataset
    daily_demand_forecasting_orders = fetch_ucirepo(id=409)

    # data (as pandas dataframes)
    X = daily_demand_forecasting_orders.data.features
    y = daily_demand_forecasting_orders.data.targets

    df = pd.concat([X, y], axis=1)
    print("First 5 rows of dataset:")
    print(df.head())

    # 2. PREPROCESSING
    # Check for null or NA values (result = 0)
    print("Missing values:")
    print(X.isna().sum())

    # Remove duplicate rows
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"\nRemoved {before - after} duplicate rows")

    # Convert categorical variables to numerical variables (N/A)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    print(X_scaled.describe())

    # Correction Matrix
    target_col = y.columns[0]
    corr_matrix = df.corr().round(2).sort_values(by=target_col, ascending=False)

    # Plot as a heatmap
    sns.heatmap(data=corr_matrix, annot=True)
    plt.tight_layout()
    plt.show()

    # Remove (four features with correction < 0.3)
    drop_cols = ["Fiscal sector orders",
                 "Banking orders (3)",
                 "Week of the month",
                 "Orders from the traffic controller sector"
                 ]

    X_scaled = X_scaled.drop(columns=drop_cols)
    print(X_scaled.head())

    # 3. SPLIT DATA: TRAINING VS TESTING (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=5)
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)

    # 4. APPLY MODEL
    # Train model
    lr_gd = LinearRegressionGD(lr=0.01, n_iter=5000, verbose=True)
    lr_gd.fit(X_train, y_train)

    # Print coefficients and intercept
    print("Coefficients:", lr_gd.w.ravel())
    print("Intercept:", lr_gd.b)

    # Predictions
    y_train_pred = lr_gd.predict(X_train)
    y_test_pred = lr_gd.predict(X_test)

    # 5. REPORT TEST & TRAIN DATASET ERROR VALUES
    # Evaluate Training Metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    train_ev = explained_variance_score(y_train, y_train_pred)

    # Evaluate Testing Metrics
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_ev = explained_variance_score(y_test, y_test_pred)

    # Compare Training and Testing Metrics
    print("\n--- Model Evaluation ---")
    print(f"{'':<10} {'MSE':<15} {'MAE':<15} {'R2':<15} {'EV':<15}")
    print(f"Train     {train_mse:<15.4f} {train_mae:<15.4f} {train_r2:<15.4f} {train_ev:<15.4f}")
    print(f"Test      {test_mse:<15.4f} {test_mae:<15.4f} {test_r2:<15.4f} {test_ev:<15.4f}")

    # Data visualizations
    plotter = RegressionPlotter()
    plotter.plot_mse_history(lr_gd.history)
    plotter.plot_predicted_vs_actual(y_train, y_train_pred, "Train")
    plotter.plot_predicted_vs_actual(y_test, y_test_pred, "Test")
    plotter.plot_residuals(y_train, y_train_pred, "Train")
    plotter.plot_residuals(y_test, y_test_pred, "Test")


if __name__ == '__main__':
    main()
