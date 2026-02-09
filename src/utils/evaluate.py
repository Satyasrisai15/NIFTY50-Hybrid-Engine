import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.models.hybrid import HybridForecaster
from src.data.dataset import StockDataset, PROCESSED_DIR
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

SEQ_LENGTH = 120
FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume", "RSI", "MACD"]

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def load_model(ticker: str) -> Optional[Tuple[Any, dict]]:
    model_path = MODELS_DIR / f"{ticker}_model.pt"
    
    if not model_path.exists():
        return None
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        model = HybridForecaster(
            input_size=len(FEATURE_COLUMNS),
            seq_length=SEQ_LENGTH,
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        scaler_params = checkpoint.get('scaler_params', None)
        
        return model, scaler_params
    except Exception as e:
        print(f"  Error loading model for {ticker}: {e}")
        return None


def evaluate_stock(ticker: str, test_ratio: float = 0.2) -> Optional[Dict[str, Any]]:
    result = load_model(ticker)
    if result is None:
        return None
    
    model, scaler_params = result
    
    try:
        dataset = StockDataset(
            data_dir=PROCESSED_DIR,
            seq_length=SEQ_LENGTH,
            tickers=[ticker],
            train=True,
        )
    except Exception as e:
        print(f"  Error loading data for {ticker}: {e}")
        return None
    
    if len(dataset) < 100:
        return None
    
    n_test = int(len(dataset) * test_ratio)
    test_indices = list(range(len(dataset) - n_test, len(dataset)))
    
    all_preds_3m = []
    all_preds_1y = []
    all_preds_3y = []
    all_actual_3m = []
    all_actual_1y = []
    all_actual_3y = []
    
    with torch.no_grad():
        for idx in test_indices:
            x, y = dataset[idx]
            x = x.unsqueeze(0).to(device)
            
            pred_3m, pred_1y, pred_3y = model(x)
            
            all_preds_3m.append(pred_3m.item())
            all_preds_1y.append(pred_1y.item())
            all_preds_3y.append(pred_3y.item())
            all_actual_3m.append(y[0].item())
            all_actual_1y.append(y[1].item())
            all_actual_3y.append(y[2].item())
    
    metrics_3m = calculate_metrics(np.array(all_actual_3m), np.array(all_preds_3m))
    metrics_1y = calculate_metrics(np.array(all_actual_1y), np.array(all_preds_1y))
    metrics_3y = calculate_metrics(np.array(all_actual_3y), np.array(all_preds_3y))
    
    return {
        'Ticker': ticker,
        'Test_Samples': n_test,
        'MSE_3M': metrics_3m['MSE'],
        'RMSE_3M': metrics_3m['RMSE'],
        'MAE_3M': metrics_3m['MAE'],
        'R2_3M': metrics_3m['R2'],
        'DirAcc_3M': metrics_3m['Directional_Accuracy'],
        'MSE_1Y': metrics_1y['MSE'],
        'RMSE_1Y': metrics_1y['RMSE'],
        'MAE_1Y': metrics_1y['MAE'],
        'R2_1Y': metrics_1y['R2'],
        'DirAcc_1Y': metrics_1y['Directional_Accuracy'],
        'MSE_3Y': metrics_3y['MSE'],
        'RMSE_3Y': metrics_3y['RMSE'],
        'MAE_3Y': metrics_3y['MAE'],
        'R2_3Y': metrics_3y['R2'],
        'DirAcc_3Y': metrics_3y['Directional_Accuracy'],
        'predictions': {
            '3M': {'actual': all_actual_3m, 'predicted': all_preds_3m},
            '1Y': {'actual': all_actual_1y, 'predicted': all_preds_1y},
            '3Y': {'actual': all_actual_3y, 'predicted': all_preds_3y},
        }
    }


def evaluate_all_stocks():
    print("=" * 70)
    print("NIFTY50 Stock Predictor - Model Evaluation")
    print("=" * 70)
    print(f"\nDevice: {device}")
    print(f"Models directory: {MODELS_DIR}")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    available_models = list(MODELS_DIR.glob("*_model.pt"))
    print(f"Found {len(available_models)} trained models")
    
    if len(available_models) == 0:
        print("\n❌ No models found!")
        return None
    
    print(f"\n{'='*70}")
    print("Evaluating models on test data (last 20% of each stock)...")
    print(f"{'='*70}\n")
    
    results = []
    all_predictions = {}
    
    for ticker in tqdm(NIFTY50_TICKERS, desc="Evaluating", unit="stock"):
        eval_result = evaluate_stock(ticker)
        
        if eval_result:
            all_predictions[ticker] = eval_result.pop('predictions')
            results.append(eval_result)
    
    if not results:
        print("\n❌ No evaluation results!")
        return None
    
    df_results = pd.DataFrame(results)
    
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE!")
    print(f"{'='*70}")
    
    print(f"\n✅ Successfully evaluated: {len(results)} stocks")
    
    print(f"\n{'='*70}")
    print("AVERAGE METRICS ACROSS ALL STOCKS")
    print(f"{'='*70}")
    
    print(f"\n{'─'*70}")
    print(f"{'Metric':<25} {'3-Month':<15} {'1-Year':<15} {'3-Year':<15}")
    print(f"{'─'*70}")
    print(f"{'MSE':<25} {df_results['MSE_3M'].mean():<15.6f} {df_results['MSE_1Y'].mean():<15.6f} {df_results['MSE_3Y'].mean():<15.6f}")
    print(f"{'RMSE':<25} {df_results['RMSE_3M'].mean():<15.6f} {df_results['RMSE_1Y'].mean():<15.6f} {df_results['RMSE_3Y'].mean():<15.6f}")
    print(f"{'MAE':<25} {df_results['MAE_3M'].mean():<15.6f} {df_results['MAE_1Y'].mean():<15.6f} {df_results['MAE_3Y'].mean():<15.6f}")
    print(f"{'R² Score':<25} {df_results['R2_3M'].mean():<15.6f} {df_results['R2_1Y'].mean():<15.6f} {df_results['R2_3Y'].mean():<15.6f}")
    print(f"{'Directional Accuracy (%)':<25} {df_results['DirAcc_3M'].mean():<15.2f} {df_results['DirAcc_1Y'].mean():<15.2f} {df_results['DirAcc_3Y'].mean():<15.2f}")
    print(f"{'─'*70}")
    
    print(f"\n{'='*70}")
    print("TOP 5 STOCKS BY DIRECTIONAL ACCURACY (3-Month)")
    print(f"{'='*70}")
    top_5 = df_results.nlargest(5, 'DirAcc_3M')[['Ticker', 'DirAcc_3M', 'DirAcc_1Y', 'DirAcc_3Y', 'R2_3M']]
    print(top_5.to_string(index=False))
    
    print(f"\n{'='*70}")
    print("BOTTOM 5 STOCKS BY DIRECTIONAL ACCURACY (3-Month)")
    print(f"{'='*70}")
    bottom_5 = df_results.nsmallest(5, 'DirAcc_3M')[['Ticker', 'DirAcc_3M', 'DirAcc_1Y', 'DirAcc_3Y', 'R2_3M']]
    print(bottom_5.to_string(index=False))
    
    results_file = RESULTS_DIR / "evaluation_metrics.csv"
    df_results.to_csv(results_file, index=False)
    print(f"\n📊 Detailed results saved to: {results_file}")
    
    summary = {
        'avg_metrics': {
            '3M': {
                'MSE': df_results['MSE_3M'].mean(),
                'RMSE': df_results['RMSE_3M'].mean(),
                'MAE': df_results['MAE_3M'].mean(),
                'R2': df_results['R2_3M'].mean(),
                'Directional_Accuracy': df_results['DirAcc_3M'].mean(),
            },
            '1Y': {
                'MSE': df_results['MSE_1Y'].mean(),
                'RMSE': df_results['RMSE_1Y'].mean(),
                'MAE': df_results['MAE_1Y'].mean(),
                'R2': df_results['R2_1Y'].mean(),
                'Directional_Accuracy': df_results['DirAcc_1Y'].mean(),
            },
            '3Y': {
                'MSE': df_results['MSE_3Y'].mean(),
                'RMSE': df_results['RMSE_3Y'].mean(),
                'MAE': df_results['MAE_3Y'].mean(),
                'R2': df_results['R2_3Y'].mean(),
                'Directional_Accuracy': df_results['DirAcc_3Y'].mean(),
            },
        },
        'per_stock': df_results,
        'predictions': all_predictions,
    }
    
    return summary


if __name__ == "__main__":
    summary = evaluate_all_stocks()

