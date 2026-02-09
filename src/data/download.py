import os
from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf
from tqdm import tqdm

NIFTY50_TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "HINDUNILVR.NS",
    "ITC.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "BAJFINANCE.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "HCLTECH.NS",
    "AXISBANK.NS",
    "ASIANPAINT.NS",
    "MARUTI.NS",
    "SUNPHARMA.NS",
    "TITAN.NS",
    "ULTRACEMCO.NS",
    "BAJAJFINSV.NS",
    "WIPRO.NS",
    "ONGC.NS",
    "NTPC.NS",
    "JSWSTEEL.NS",
    "POWERGRID.NS",
    "M&M.NS",
    "TMPV.NS",
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "COALINDIA.NS",
    "TATASTEEL.NS",
    "HINDALCO.NS",
    "SBILIFE.NS",
    "BAJAJ-AUTO.NS",
    "GRASIM.NS",
    "DIVISLAB.NS",
    "BRITANNIA.NS",
    "CIPLA.NS",
    "TECHM.NS",
    "NESTLEIND.NS",
    "APOLLOHOSP.NS",
    "HEROMOTOCO.NS",
    "INDUSINDBK.NS",
    "EICHERMOT.NS",
    "DRREDDY.NS",
    "TATACONSUM.NS",
    "BPCL.NS",
    "HDFCLIFE.NS",
    "UPL.NS",
    "LTIM.NS",
]

START_DATE = "2005-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")

def download_stock_data(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end)
        
        if df.empty:
            return None
        
        return df
    except Exception as e:
        print(f"  Error downloading {ticker}: {e}")
        return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Downloading NIFTY50 stock data from {START_DATE} to {END_DATE}")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    successful = 0
    failed = []
    
    for ticker in tqdm(NIFTY50_TICKERS, desc="Downloading stocks", unit="stock"):
        df = download_stock_data(ticker, START_DATE, END_DATE)
        
        if df is None or df.empty:
            print(f"  ⚠️  Warning: No data available for {ticker}")
            failed.append(ticker)
            continue
        
        filename = ticker.replace(".NS", "") + ".csv"
        filepath = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(filepath)
        successful += 1
    
    print(f"\n{'='*50}")
    print(f"Download complete!")
    print(f"  ✅ Successfully downloaded: {successful}/{len(NIFTY50_TICKERS)} stocks")
    
    if failed:
        print(f"  ❌ Failed to download: {len(failed)} stocks")
        print(f"     Missing tickers: {', '.join(failed)}")


if __name__ == "__main__":
    main()
