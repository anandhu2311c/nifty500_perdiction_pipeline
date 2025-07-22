import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os

def compute_sharpe(returns, annualization=252):
    # Expects returns as a 1D array or Series
    mu = np.mean(returns)
    sigma = np.std(returns)
    return (mu * np.sqrt(annualization)) / (sigma + 1e-9)

def create_sharpe_forecast_features(df, window=365, horizon=30):
    features, targets, dates = [], [], []
    # Ensure correct columns and remove NaN returns
    df = df.copy()
    if 'Daily_Return' not in df or 'Volume' not in df:
        raise ValueError("Dataframe must contain 'Daily_Return' and 'Volume' columns.")
    df = df.dropna(subset=['Daily_Return'])
    for i in range(window, len(df) - horizon):
        past = df.iloc[i-window:i]
        future = df.iloc[i:i+horizon]
        feat = {
            'rolling_sharpe_30': compute_sharpe(past['Daily_Return'][-30:]),
            'vol_30': past['Daily_Return'][-30:].std() * np.sqrt(252),
            'vol_60': past['Daily_Return'][-60:].std() * np.sqrt(252),
            'vol_90': past['Daily_Return'][-90:].std() * np.sqrt(252),
            'return_7d_mean': past['Daily_Return'][-7:].mean(),
            'return_30d_mean': past['Daily_Return'][-30:].mean(),
            'return_7d_std': past['Daily_Return'][-7:].std(),
            'volume_mean_30': past['Volume'][-30:].mean(),
            'volume_ratio': (past['Volume'][-1] / (past['Volume'][-30:].mean() + 1e-6)),
        }
        # Target: Sharpe over next 30 days (or horizon)
        features.append(feat)
        targets.append(compute_sharpe(future['Daily_Return']))
        dates.append(df.index[i])
    X = pd.DataFrame(features, index=dates)
    y = pd.Series(targets, index=dates)
    return X, y

def train_and_save_model(ticker_csv, model_path, window=365, horizon=30, test_size=0.2):
    print(f"Training Sharpe forecaster for {ticker_csv} ...")
    df = pd.read_csv(ticker_csv, parse_dates=['Date'], index_col='Date')
    X, y = create_sharpe_forecast_features(df, window, horizon)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    model = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"R2 Score: {r2:.3f} | MAE: {mae:.3f}")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # Change filename and output path as needed
    train_and_save_model(
        'data/processed/INFY.csv', 
        'models/next30_sharpe_gbm.pkl',
        window=365,
        horizon=30,
        test_size=0.2
    )
