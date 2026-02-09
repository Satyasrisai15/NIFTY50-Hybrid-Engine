import os
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from tqdm import tqdm

from src.models.hybrid import HybridForecaster

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
EVALUATION_DIR = BASE_DIR / "results"

SEQ_LENGTH = 120
FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

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


def download_data_for_date(ticker: str, prediction_date: datetime) -> Optional[pd.DataFrame]:
    try:
        end_date = prediction_date
        start_date = end_date - timedelta(days=200)
        
        stock = yf.Ticker(f"{ticker}.NS")
        df = stock.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
        
        if df.empty:
            return None
        
        df = df[df.index.date <= prediction_date.date()]
        
        # If we have short data (e.g. IPO/Demerger), pad it to meet SEQ_LENGTH
        if len(df) < SEQ_LENGTH:
            # Need at least one row to pad
            if len(df) == 0:
                print(f"  ⚠️  Data empty for {ticker} by date {prediction_date.date()}")
                return None
            
            missing = SEQ_LENGTH - len(df)
            print(f"  ⚠️  Short history for {ticker} (Length={len(df)}). Padding {missing} rows.")
            
            # Create padding rows by repeating the first row
            first_row = df.iloc[[0]]
            padding = pd.concat([first_row] * missing, ignore_index=True)
            # We need to preserve dataframe columns and preferably index structure, 
            # though index will be dummy.
            
            # Simple concat: put padding on TOP (past).
            # But yfinance dataframe has DateTime Index.
            # We can just reset index for manipulation or create dummy dates.
            # Let's just concat values logicwise, since we only extract features later.
            
            # Actually, `predict_stock` extracts features: `df[FEATURE_COLUMNS].values`
            # So index manipulation isn't strictly necessary if it returns a DF with correct columns.
            
            df_pad = pd.concat([padding, df.reset_index(drop=True)], ignore_index=True)
            # Reattach columns
            for col in df.columns:
                 # Ensure types match
                 pass
            
            return df_pad

        return df.tail(SEQ_LENGTH)
    except Exception as e:
        print(f"  Error downloading {ticker}: {e}")
        return None


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


def normalize_data(data: np.ndarray, scaler_params: dict) -> np.ndarray:
    min_vals = scaler_params['min']
    max_vals = scaler_params['max']
    
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    
    return (data - min_vals) / range_vals


def predict_stock(ticker: str, prediction_date: datetime) -> Optional[Dict[str, Any]]:
    result = load_model(ticker)
    if result is None:
        print(f"  ⚠️  No model found for {ticker}")
        return None
    
    model, scaler_params = result
    
    df = download_data_for_date(ticker, prediction_date)
    if df is None:
        print(f"  ⚠️  Could not download data for {ticker}")
        return None
    
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"  ⚠️  Missing columns for {ticker}: {missing_cols}")
        return None
    
    features = df[FEATURE_COLUMNS].values.astype(np.float32)
    
    if scaler_params:
        features = normalize_data(features, scaler_params)
    
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_3m, pred_1y, pred_3y = model(x)
    
    return {
        'Ticker': ticker,
        'Prediction_Date': prediction_date.strftime("%Y-%m-%d"),
        'Predicted_Return_3M': pred_3m.item(),
        'Predicted_Return_1Y': pred_1y.item(),
        'Predicted_Return_3Y': pred_3y.item(),
    }


def main():
    parser = argparse.ArgumentParser(description='NIFTY50 Stock Return Predictor')
    parser.add_argument('--date', type=str, default=None,
                        help='Prediction start date (YYYY-MM-DD). Default: today')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV filename. Default: evaluation_results/final_report_YYYYMMDD.csv')
    args = parser.parse_args()
    
    if args.date:
        prediction_date = datetime.strptime(args.date, "%Y-%m-%d")
    else:
        prediction_date = datetime.today()
    
    if args.output:
        # If user provides path, check if it's absolute or relative to cwd.
        # We can just use it as is if it has path separators, or put in evaluation_results if just filename.
        # But simple behavior: allow user full control if specified.
        output_file = Path(args.output)
    else:
        EVALUATION_DIR.mkdir(exist_ok=True, parents=True)
        output_file = EVALUATION_DIR / f"final_report_{prediction_date.strftime('%Y%m%d')}.csv"
    
    date_3m = prediction_date + timedelta(days=90)
    date_1y = prediction_date + timedelta(days=365)
    date_3y = prediction_date + timedelta(days=365*3)
    
    print("=" * 70)
    print("NIFTY50 Stock Predictor - Generating Predictions")
    print("=" * 70)
    print(f"\nDevice: {device}")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Output file: {output_file}")
    
    print(f"\n{'='*70}")
    print("PREDICTION TIMELINE")
    print(f"{'='*70}")
    print(f"  Prediction FROM:     {prediction_date.strftime('%B %d, %Y')}")
    print(f"  3-Month target:      {date_3m.strftime('%B %d, %Y')} (~{date_3m.strftime('%b %Y')})")
    print(f"  1-Year target:       {date_1y.strftime('%B %d, %Y')} (~{date_1y.strftime('%b %Y')})")
    print(f"  3-Year target:       {date_3y.strftime('%B %d, %Y')} (~{date_3y.strftime('%b %Y')})")
    
    available_models = list(MODELS_DIR.glob("*_model.pt"))
    print(f"\nFound {len(available_models)} trained models")
    
    if len(available_models) == 0:
        print("\n❌ No models found! Please train models first.")
        return
    
    print(f"\n{'='*70}")
    print("Generating predictions for all stocks...")
    print(f"{'='*70}\n")
    
    results = []
    successful = 0
    failed = []
    
    for ticker in tqdm(NIFTY50_TICKERS, desc="Predicting", unit="stock"):
        prediction = predict_stock(ticker, prediction_date)
        
        if prediction:
            results.append(prediction)
            successful += 1
        else:
            failed.append(ticker)
    
    if not results:
        print("\n❌ No predictions generated!")
        return
    
    df_results = pd.DataFrame(results)
    
    df_results = df_results.sort_values('Predicted_Return_3M', ascending=False)
    
    df_results.to_csv(output_file, index=False)
    
    print(f"\n{'='*70}")
    print("PREDICTION COMPLETE!")
    print(f"{'='*70}")
    
    print(f"\n✅ Successfully predicted: {successful}/{len(NIFTY50_TICKERS)} stocks")
    
    if failed:
        print(f"❌ Failed: {len(failed)} stocks")
        print(f"   {', '.join(failed)}")
    
    print(f"\n📊 Results saved to: {output_file}")
    
    print(f"\n{'='*70}")
    print(f"TOP 10 STOCKS BY 3-MONTH RETURN (by {date_3m.strftime('%b %Y')})")
    print(f"{'='*70}")
    print(df_results[['Ticker', 'Predicted_Return_3M', 'Predicted_Return_1Y', 'Predicted_Return_3Y']].head(10).to_string(index=False))
    
    print(f"\n{'='*70}")
    print(f"TOP 10 STOCKS BY 1-YEAR RETURN (by {date_1y.strftime('%b %Y')})")
    print(f"{'='*70}")
    df_1y = df_results.sort_values('Predicted_Return_1Y', ascending=False)
    print(df_1y[['Ticker', 'Predicted_Return_3M', 'Predicted_Return_1Y', 'Predicted_Return_3Y']].head(10).to_string(index=False))
    
    print(f"\n{'='*70}")
    print(f"TOP 10 STOCKS BY 3-YEAR RETURN (by {date_3y.strftime('%b %Y')})")
    print(f"{'='*70}")
    df_3y = df_results.sort_values('Predicted_Return_3Y', ascending=False)
    print(df_3y[['Ticker', 'Predicted_Return_3M', 'Predicted_Return_1Y', 'Predicted_Return_3Y']].head(10).to_string(index=False))
    
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    
    for col, period in [('Predicted_Return_3M', '3-Month'), 
                        ('Predicted_Return_1Y', '1-Year'),
                        ('Predicted_Return_3Y', '3-Year')]:
        print(f"\n{period} Returns:")
        print(f"  Mean:   {df_results[col].mean()*100:+.2f}%")
        print(f"  Std:    {df_results[col].std()*100:.2f}%")
        print(f"  Min:    {df_results[col].min()*100:+.2f}%")
        print(f"  Max:    {df_results[col].max()*100:+.2f}%")


if __name__ == "__main__":
    main()
