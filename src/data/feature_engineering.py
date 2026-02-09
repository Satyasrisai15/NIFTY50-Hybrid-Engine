import os
from pathlib import Path
from typing import Union
import pandas as pd
from tqdm import tqdm

DAYS_3M = 63
DAYS_1Y = 252
DAYS_3Y = 756

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def load_stock_csv(filepath: Union[str, Path]) -> pd.DataFrame:
    return pd.read_csv(filepath, index_col=0, parse_dates=True)

def create_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Target_3M"] = df["Close"].shift(-DAYS_3M)
    df["Target_1Y"] = df["Close"].shift(-DAYS_1Y)
    df["Target_3Y"] = df["Close"].shift(-DAYS_3Y)
    return df

def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Return_3M"] = (df["Target_3M"] - df["Close"]) / df["Close"]
    df["Return_1Y"] = (df["Target_1Y"] - df["Close"]) / df["Close"]
    df["Return_3Y"] = (df["Target_3Y"] - df["Close"]) / df["Close"]
    return df

def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def process_stock(input_path: Union[str, Path], output_path: Union[str, Path]) -> bool:
    try:
        df = load_stock_csv(input_path)
        if "Close" not in df.columns:
            print(f"  ⚠️ Warning: 'Close' missing in {input_path}")
            return False
        
        # Add Technical Indicators
        df["RSI"] = calculate_rsi(df["Close"])
        macd, s = calculate_macd(df["Close"])
        df["MACD"] = macd
        df["MACD_Signal"] = s
        df = df.bfill().ffill()  # Clean NaNs
        
        df = create_target_columns(df)
        df = calculate_returns(df)
        df.to_csv(output_path)
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def process_all_stocks():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    csv_files = list(RAW_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {RAW_DIR}")
        return
    print(f"Processing {len(csv_files)} files...")
    successful = 0
    for f in tqdm(csv_files):
        if process_stock(f, PROCESSED_DIR / f.name):
            successful += 1
    print(f"\n✅ Successfully processed: {successful}/{len(csv_files)} stocks")

if __name__ == "__main__":
    process_all_stocks()
