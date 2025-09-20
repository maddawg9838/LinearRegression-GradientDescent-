import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import seaborn as sns

matplotlib.use("TkAgg")


# Linear Regression using Scikit-learn's SGDRegressor with gradient descent.
class LinearRegressionSK:
    def __init__(self, max_iter=5000, tol=1e-3, random_state=5):
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.model = SGDRegressor(max_iter=self.max_iter, tol=self.tol, random_state=self.random_state)

    # Scale features and drop low-correlated columns
    def preprocess(self, X, drop_columns=None):
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        if drop_columns:
            X_scaled = X_scaled.drop(columns=drop_columns)
        self.X_scaled = X_scaled
        return self.X_scaled

    # Split dataset into training and testing
    def train_test_split(self, X, y, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    # Fit model
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train.values.ravel())

    # Predict target values
    def predict(self, X):
        return self.model.predict(X)

    # 5. REPORT TEST & TRAIN DATASET ERROR VALUES
    def evaluate(self, X, y, dataset_name="Dataset"):
        """Compute and print evaluation metrics."""
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        ev = explained_variance_score(y, y_pred)
        print(f"\n--- {dataset_name} Evaluation ---")
        print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, EV: {ev:.4f}")
        return {"MSE": mse, "MAE": mae, "R2": r2, "EV": ev}

    def cross_validate(self, X, y, cv_splits=5):
        """Perform cross-validation to check generalization."""
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
        cv_mse = cross_val_score(self.model, X, y.values.ravel(), cv=kf, scoring="neg_mean_squared_error")
        cv_r2 = cross_val_score(self.model, X, y.values.ravel(), cv=kf, scoring="r2")
        print("\n--- Cross Validation ---")
        print(f"Average MSE: {-cv_mse.mean():.4f} (± {cv_mse.std():.4f})")
        print(f"Average R²:  {cv_r2.mean():.4f} (± {cv_r2.std():.4f})")
        return {"CV_MSE": -cv_mse.mean(), "CV_R2": cv_r2.mean()}


# Plotting class
class RegressionPlotter:
    def plot_predicted_vs_actual(self, y_true, y_pred, dataset_name="Dataset"):
        plt.figure(figsize=(6, 6))
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()
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
    daily_demand_forecasting_orders = fetch_ucirepo(id=409)
    X = daily_demand_forecasting_orders.data.features
    y = daily_demand_forecasting_orders.data.targets

    # 2. PREPROCESSING
    # Remove low-correlated columns
    drop_cols = ["Fiscal sector orders", "Banking orders (3)", "Week of the month",
                 "Orders from the traffic controller sector"]

    lr_sk = LinearRegressionSK(max_iter=5000, tol=1e-3, random_state=5)
    X_scaled = lr_sk.preprocess(X, drop_columns=drop_cols)

    # 3. SPLIT DATA: TRAINING VS TESTING
    X_train, X_test, y_train, y_test = lr_sk.train_test_split(X_scaled, y)

    # 4. TRAIN MODEL
    lr_sk.fit(X_train, y_train)
    print("Coefficients:", lr_sk.model.coef_)
    print("Intercept:", lr_sk.model.intercept_)

    # 5. EVALUATE MODEL
    lr_sk.evaluate(X_train, y_train, "Train")
    lr_sk.evaluate(X_test, y_test, "Test")

    # Cross-validation
    lr_sk.cross_validate(X_scaled, y)

    # Data Visualizations
    plotter = RegressionPlotter()
    y_train_pred = lr_sk.predict(X_train)
    y_test_pred = lr_sk.predict(X_test)

    plotter.plot_predicted_vs_actual(y_train.values.ravel(), y_train_pred, "Train")
    plotter.plot_predicted_vs_actual(y_test.values.ravel(), y_test_pred, "Test")

    plotter.plot_residuals(y_train.values.ravel(), y_train_pred, "Train")
    plotter.plot_residuals(y_test.values.ravel(), y_test_pred, "Test")


if __name__ == "__main__":
    main()
