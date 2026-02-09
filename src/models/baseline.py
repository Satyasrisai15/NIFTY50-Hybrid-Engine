import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

import sys

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.dataset import StockDataset, PROCESSED_DIR, FEATURE_COLUMNS, TARGET_COLUMNS
from src.models.hybrid import HybridForecaster

BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

SEQ_LENGTH = 120

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NIFTY50_TICKERS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "BAJFINANCE",
    "KOTAKBANK", "LT", "HCLTECH", "AXISBANK", "ASIANPAINT",
    "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "BAJAJFINSV",
    "WIPRO", "ONGC", "NTPC", "JSWSTEEL", "POWERGRID",
    "M&M", "TMPV", "ADANIENT", "ADANIPORTS", "COALINDIA",
    "TATASTEEL", "HINDALCO", "SBILIFE", "BAJAJ-AUTO", "GRASIM",
    "DIVISLAB", "BRITANNIA", "CIPLA", "TECHM", "NESTLEIND",
    "APOLLOHOSP", "HEROMOTOCO", "INDUSINDBK", "EICHERMOT", "DRREDDY",
    "TATACONSUM", "BPCL", "HDFCLIFE", "UPL", "LTIM",
]


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    if len(y_true) > 1 and np.var(y_true) > 0:
        r2 = r2_score(y_true, y_pred)
    else:
        r2 = 0.0
    
    true_direction = (y_true > 0).astype(int)
    pred_direction = (y_pred > 0).astype(int)
    directional_accuracy = (true_direction == pred_direction).mean() * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Directional_Accuracy': directional_accuracy
    }


class SimpleLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=1, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


def moving_average_baseline(df: pd.DataFrame, window: int = 20) -> dict:
    close = df['Close'].values
    returns_3m = df['Return_3M'].values
    returns_1y = df['Return_1Y'].values
    returns_3y = df['Return_3Y'].values
    
    ma = pd.Series(close).rolling(window=window).mean().values
    
    pred_direction = np.where(close > ma, 1, -1)
    
    avg_positive_return_3m = np.nanmean(returns_3m[returns_3m > 0]) if np.any(returns_3m > 0) else 0.05
    avg_negative_return_3m = np.nanmean(returns_3m[returns_3m < 0]) if np.any(returns_3m < 0) else -0.05
    avg_positive_return_1y = np.nanmean(returns_1y[returns_1y > 0]) if np.any(returns_1y > 0) else 0.15
    avg_negative_return_1y = np.nanmean(returns_1y[returns_1y < 0]) if np.any(returns_1y < 0) else -0.15
    avg_positive_return_3y = np.nanmean(returns_3y[returns_3y > 0]) if np.any(returns_3y > 0) else 0.50
    avg_negative_return_3y = np.nanmean(returns_3y[returns_3y < 0]) if np.any(returns_3y < 0) else -0.30
    
    pred_3m = np.where(pred_direction > 0, avg_positive_return_3m, avg_negative_return_3m)
    pred_1y = np.where(pred_direction > 0, avg_positive_return_1y, avg_negative_return_1y)
    pred_3y = np.where(pred_direction > 0, avg_positive_return_3y, avg_negative_return_3y)
    
    return {
        'pred_3m': pred_3m,
        'pred_1y': pred_1y,
        'pred_3y': pred_3y,
    }


def historical_mean_baseline(df: pd.DataFrame) -> dict:
    mean_3m = df['Return_3M'].mean()
    mean_1y = df['Return_1Y'].mean()
    mean_3y = df['Return_3Y'].mean()
    
    n = len(df)
    
    return {
        'pred_3m': np.full(n, mean_3m),
        'pred_1y': np.full(n, mean_1y),
        'pred_3y': np.full(n, mean_3y),
    }


def linear_regression_baseline(X_train, y_train, X_test) -> np.ndarray:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.predict(X_test)


def random_forest_baseline(X_train, y_train, X_test) -> np.ndarray:
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def train_simple_lstm(X_train, y_train, X_test, epochs=5):
    input_size = X_train.shape[2]
    model = SimpleLSTM(input_size=input_size, hidden_size=64, num_layers=1, output_size=1)
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_t).cpu().numpy().flatten()
    
    return predictions


from typing import Optional, Union, Dict, Any

def evaluate_baselines_for_stock(ticker: str, test_ratio: float = 0.2) -> Optional[Dict[str, Any]]:
    filepath = PROCESSED_DIR / f"{ticker}.csv"
    if not filepath.exists():
        return None
    
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    required_cols = FEATURE_COLUMNS + TARGET_COLUMNS
    if not all(col in df.columns for col in required_cols):
        return None
    
    df = df.dropna(subset=required_cols)
    
    if len(df) < 500:
        return None
    
    n_test = int(len(df) * test_ratio)
    train_df = df.iloc[:-n_test]
    test_df = df.iloc[-n_test:]
    
    actual_3m = test_df['Return_3M'].values
    actual_1y = test_df['Return_1Y'].values
    actual_3y = test_df['Return_3Y'].values
    
    results = {'Ticker': ticker, 'Test_Samples': n_test}
    
    hist_mean = historical_mean_baseline(train_df)
    results['HistMean_MSE_3M'] = mean_squared_error(actual_3m, np.full(n_test, hist_mean['pred_3m'].mean()))
    results['HistMean_DirAcc_3M'] = calculate_metrics(actual_3m, np.full(n_test, hist_mean['pred_3m'].mean()))['Directional_Accuracy']
    results['HistMean_MSE_1Y'] = mean_squared_error(actual_1y, np.full(n_test, hist_mean['pred_1y'].mean()))
    results['HistMean_DirAcc_1Y'] = calculate_metrics(actual_1y, np.full(n_test, hist_mean['pred_1y'].mean()))['Directional_Accuracy']
    
    ma_preds = moving_average_baseline(test_df)
    results['MA_MSE_3M'] = mean_squared_error(actual_3m, ma_preds['pred_3m'])
    results['MA_DirAcc_3M'] = calculate_metrics(actual_3m, ma_preds['pred_3m'])['Directional_Accuracy']
    results['MA_MSE_1Y'] = mean_squared_error(actual_1y, ma_preds['pred_1y'])
    results['MA_DirAcc_1Y'] = calculate_metrics(actual_1y, ma_preds['pred_1y'])['Directional_Accuracy']
    
    X_train_flat = train_df[FEATURE_COLUMNS].values
    X_test_flat = test_df[FEATURE_COLUMNS].values
    
    lr_pred_3m = linear_regression_baseline(X_train_flat, train_df['Return_3M'].values, X_test_flat)
    results['LR_MSE_3M'] = mean_squared_error(actual_3m, lr_pred_3m)
    results['LR_DirAcc_3M'] = calculate_metrics(actual_3m, lr_pred_3m)['Directional_Accuracy']
    
    lr_pred_1y = linear_regression_baseline(X_train_flat, train_df['Return_1Y'].values, X_test_flat)
    results['LR_MSE_1Y'] = mean_squared_error(actual_1y, lr_pred_1y)
    results['LR_DirAcc_1Y'] = calculate_metrics(actual_1y, lr_pred_1y)['Directional_Accuracy']
    
    rf_pred_3m = random_forest_baseline(X_train_flat, train_df['Return_3M'].values, X_test_flat)
    results['RF_MSE_3M'] = mean_squared_error(actual_3m, rf_pred_3m)
    results['RF_DirAcc_3M'] = calculate_metrics(actual_3m, rf_pred_3m)['Directional_Accuracy']
    
    rf_pred_1y = random_forest_baseline(X_train_flat, train_df['Return_1Y'].values, X_test_flat)
    results['RF_MSE_1Y'] = mean_squared_error(actual_1y, rf_pred_1y)
    results['RF_DirAcc_1Y'] = calculate_metrics(actual_1y, rf_pred_1y)['Directional_Accuracy']
    
    try:
        dataset = StockDataset(
            data_dir=PROCESSED_DIR,
            seq_length=SEQ_LENGTH,
            tickers=[ticker],
            train=True,
        )
        
        n_test_seq = int(len(dataset) * test_ratio)
        train_indices = list(range(len(dataset) - n_test_seq))
        test_indices = list(range(len(dataset) - n_test_seq, len(dataset)))
        
        X_train_seq = np.array([dataset.X[i] for i in train_indices])
        y_train_3m = np.array([dataset.Y[i][0] for i in train_indices])
        y_train_1y = np.array([dataset.Y[i][1] for i in train_indices])
        X_test_seq = np.array([dataset.X[i] for i in test_indices])
        y_test_3m = np.array([dataset.Y[i][0] for i in test_indices])
        y_test_1y = np.array([dataset.Y[i][1] for i in test_indices])
        
        lstm_pred_3m = train_simple_lstm(X_train_seq, y_train_3m, X_test_seq, epochs=5)
        results['SimpleLSTM_MSE_3M'] = mean_squared_error(y_test_3m, lstm_pred_3m)
        results['SimpleLSTM_DirAcc_3M'] = calculate_metrics(y_test_3m, lstm_pred_3m)['Directional_Accuracy']
        
        lstm_pred_1y = train_simple_lstm(X_train_seq, y_train_1y, X_test_seq, epochs=5)
        results['SimpleLSTM_MSE_1Y'] = mean_squared_error(y_test_1y, lstm_pred_1y)
        results['SimpleLSTM_DirAcc_1Y'] = calculate_metrics(y_test_1y, lstm_pred_1y)['Directional_Accuracy']
        
        model_path = MODELS_DIR / f"{ticker}_model.pt"
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            hybrid_model = HybridForecaster(
                input_size=len(FEATURE_COLUMNS),
                seq_length=SEQ_LENGTH,
            ).to(device)
            hybrid_model.load_state_dict(checkpoint['model_state_dict'])
            hybrid_model.eval()
            
            hybrid_preds_3m = []
            hybrid_preds_1y = []
            
            with torch.no_grad():
                for idx in test_indices:
                    x, _ = dataset[idx]
                    x = x.unsqueeze(0).to(device)
                    pred_3m, pred_1y, _ = hybrid_model(x)
                    hybrid_preds_3m.append(pred_3m.item())
                    hybrid_preds_1y.append(pred_1y.item())
            
            results['Hybrid_MSE_3M'] = mean_squared_error(y_test_3m, hybrid_preds_3m)
            results['Hybrid_DirAcc_3M'] = calculate_metrics(y_test_3m, np.array(hybrid_preds_3m))['Directional_Accuracy']
            results['Hybrid_MSE_1Y'] = mean_squared_error(y_test_1y, hybrid_preds_1y)
            results['Hybrid_DirAcc_1Y'] = calculate_metrics(y_test_1y, np.array(hybrid_preds_1y))['Directional_Accuracy']
        
    except Exception as e:
        print(f"  Error with sequence models for {ticker}: {e}")
    
    return results


def compare_all_models():
    print("=" * 70)
    print("BASELINE MODEL COMPARISON")
    print("=" * 70)
    print(f"\nDevice: {device}")
    print(f"\nModels being compared:")
    print("  1. Historical Mean (Naive baseline)")
    print("  2. Moving Average (20-day)")
    print("  3. Linear Regression")
    print("  4. Random Forest")
    print("  5. Simple LSTM (1 layer)")
    print("  6. HybridForecaster (CNN + BiLSTM + Transformer) [Ours]")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("Evaluating all models...")
    print(f"{'='*70}\n")
    
    results = []
    
    for ticker in tqdm(NIFTY50_TICKERS, desc="Comparing models", unit="stock"):
        result = evaluate_baselines_for_stock(ticker)
        if result:
            results.append(result)
    
    if not results:
        print("\n❌ No results!")
        return
    
    df_results = pd.DataFrame(results)
    
    print(f"\n{'='*70}")
    print("MODEL COMPARISON RESULTS (3-Month Prediction)")
    print(f"{'='*70}")
    
    models_3m = [
        ('Historical Mean', 'HistMean_MSE_3M', 'HistMean_DirAcc_3M'),
        ('Moving Average', 'MA_MSE_3M', 'MA_DirAcc_3M'),
        ('Linear Regression', 'LR_MSE_3M', 'LR_DirAcc_3M'),
        ('Random Forest', 'RF_MSE_3M', 'RF_DirAcc_3M'),
        ('Simple LSTM', 'SimpleLSTM_MSE_3M', 'SimpleLSTM_DirAcc_3M'),
        ('HybridForecaster', 'Hybrid_MSE_3M', 'Hybrid_DirAcc_3M'),
    ]
    
    print(f"\n{'Model':<25} {'Avg MSE':<15} {'Avg Dir. Acc (%)':<20}")
    print(f"{'─'*60}")
    
    comparison_data = []
    for name, mse_col, acc_col in models_3m:
        if mse_col in df_results.columns:
            avg_mse = df_results[mse_col].mean()
            avg_acc = df_results[acc_col].mean()
            print(f"{name:<25} {avg_mse:<15.6f} {avg_acc:<20.2f}")
            comparison_data.append({'Model': name, 'MSE_3M': avg_mse, 'DirAcc_3M': avg_acc})
    
    print(f"\n{'='*70}")
    print("MODEL COMPARISON RESULTS (1-Year Prediction)")
    print(f"{'='*70}")
    
    models_1y = [
        ('Historical Mean', 'HistMean_MSE_1Y', 'HistMean_DirAcc_1Y'),
        ('Moving Average', 'MA_MSE_1Y', 'MA_DirAcc_1Y'),
        ('Linear Regression', 'LR_MSE_1Y', 'LR_DirAcc_1Y'),
        ('Random Forest', 'RF_MSE_1Y', 'RF_DirAcc_1Y'),
        ('Simple LSTM', 'SimpleLSTM_MSE_1Y', 'SimpleLSTM_DirAcc_1Y'),
        ('HybridForecaster', 'Hybrid_MSE_1Y', 'Hybrid_DirAcc_1Y'),
    ]
    
    print(f"\n{'Model':<25} {'Avg MSE':<15} {'Avg Dir. Acc (%)':<20}")
    print(f"{'─'*60}")
    
    for name, mse_col, acc_col in models_1y:
        if mse_col in df_results.columns:
            avg_mse = df_results[mse_col].mean()
            avg_acc = df_results[acc_col].mean()
            print(f"{name:<25} {avg_mse:<15.6f} {avg_acc:<20.2f}")
            for item in comparison_data:
                if item['Model'] == name:
                    item['MSE_1Y'] = avg_mse
                    item['DirAcc_1Y'] = avg_acc
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_file = RESULTS_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_file, index=False)
    
    detailed_file = RESULTS_DIR / "baseline_detailed_results.csv"
    df_results.to_csv(detailed_file, index=False)
    
    print(f"\n📊 Comparison saved to: {comparison_file}")
    print(f"📊 Detailed results saved to: {detailed_file}")
    
    if 'Hybrid_DirAcc_3M' in df_results.columns:
        hybrid_acc = df_results['Hybrid_DirAcc_3M'].mean()
        best_baseline_acc = max(
            df_results['HistMean_DirAcc_3M'].mean(),
            df_results['MA_DirAcc_3M'].mean(),
            df_results['LR_DirAcc_3M'].mean(),
            df_results['RF_DirAcc_3M'].mean(),
            df_results['SimpleLSTM_DirAcc_3M'].mean() if 'SimpleLSTM_DirAcc_3M' in df_results.columns else 0
        )
        
        improvement = hybrid_acc - best_baseline_acc
        
        print(f"\n{'='*70}")
        print("CONCLUSION")
        print(f"{'='*70}")
        print(f"\nHybridForecaster Directional Accuracy: {hybrid_acc:.2f}%")
        print(f"Best Baseline Directional Accuracy: {best_baseline_acc:.2f}%")
        print(f"Improvement: {improvement:+.2f}%")
        
        if improvement > 0:
            print(f"\n✅ HybridForecaster OUTPERFORMS baselines by {improvement:.2f}%")
        else:
            print(f"\n⚠️ HybridForecaster underperforms best baseline by {abs(improvement):.2f}%")
    
    return comparison_df


if __name__ == "__main__":
    compare_all_models()

