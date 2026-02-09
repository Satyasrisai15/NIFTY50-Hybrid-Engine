import os
from pathlib import Path
from typing import Optional, Union, List, Tuple
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


BASE_DIR = Path(__file__).parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume", "RSI", "MACD"]

TARGET_COLUMNS = ["Return_3M", "Return_1Y", "Return_3Y"]


class StockDataset(Dataset):
    def __init__(
        self,
        data_dir: Union[str, Path] = PROCESSED_DIR,
        seq_length: int = 120,
        feature_cols: List[str] = FEATURE_COLUMNS,
        target_cols: List[str] = TARGET_COLUMNS,
        tickers: Optional[List[str]] = None,
        train: bool = True,
        scaler_params: Optional[dict] = None,
    ):
        self.data_dir = Path(data_dir)
        self.seq_length = seq_length
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.train = train
        
        self.X = []
        self.Y = []
        self.metadata = []
        
        self.scaler_params = scaler_params or {}
        
        self._load_data(tickers)
        
        self.X = np.array(self.X, dtype=np.float32)
        self.Y = np.array(self.Y, dtype=np.float32)
        
        self._normalize_data()
    
    def _load_data(self, tickers: Optional[List[str]] = None):
        if tickers:
            csv_files = [self.data_dir / f"{t}.csv" for t in tickers]
            csv_files = [f for f in csv_files if f.exists()]
        else:
            csv_files = list(self.data_dir.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
        
        print(f"Loading {len(csv_files)} stock files...")
        
        for csv_file in tqdm(csv_files, desc="Loading stocks", unit="stock"):
            ticker = csv_file.stem
            self._process_stock(csv_file, ticker)
        
        print(f"Total samples created: {len(self.X):,}")
    
    def _process_stock(self, filepath: Path, ticker: str):
        try:
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            
            missing_features = [c for c in self.feature_cols if c not in df.columns]
            missing_targets = [c for c in self.target_cols if c not in df.columns]
            
            if missing_features or missing_targets:
                print(f"  ⚠️  Skipping {ticker}: missing columns")
                return
            
            features = df[self.feature_cols].values
            targets = df[self.target_cols].values
            dates = df.index
            
            # Handle short sequences by padding
            if len(df) < self.seq_length:
                pad_len = self.seq_length - len(df)
                # Pad with the first value (edge padding) to preserve scale
                features_pad = np.tile(features[0], (pad_len, 1))
                features = np.vstack([features_pad, features])
                
                # For targets, we can't really pad validly, but for training mechanics we can
                # replicate the first target or use dummy. Since we need X[i] -> Y[i+seq],
                # we just need enough history to make ONE sample.
                # Actually, if we have < seq_length, we can only create samples if we pad.
                # Let's pad targets too just to keep indices aligned, though we mainly care about features.
                targets_pad = np.tile(targets[0], (pad_len, 1))
                targets = np.vstack([targets_pad, targets])
                
                # Re-calculate indices based on new padded length
                # We effectively only have valid "real" data at the end.
                # We will generate at least one sample.
            
            for i in range(len(features) - self.seq_length + 1):
                # Ensure we have at least one sample even if equals seq_length
                if i + self.seq_length > len(features):
                     break
                     
                x = features[i:i + self.seq_length]
                # Target is the one AFTER the sequence
                # In original code: targets[i + self.seq_length - 1] 
                # (Wait, original code was taking target at the END of sequence? 
                # feature_engineering.py shifts targets backwards, so target at index i IS the future return for i)
                # Let's check feature_engineering.py:
                # df["Target_3M"] = df["Close"].shift(-63)
                # So row T has features for time T, and Target_3M for time T.
                # We want input X[T-seq+1 : T] -> Y[T].
                # Standard LSTM input X_t predicts Y_t.
                # In dataset.py original:
                # x = features[i:i + self.seq_length]
                # y = targets[i + self.seq_length - 1]
                # This means we take a window of 120 steps, and the target corresponding to the LAST step (t=119).
                # This seems correct for "predict return at time T given history up to T".
                
                try:
                    y = targets[i + self.seq_length - 1]
                except IndexError:
                    continue
                
                if np.isnan(x).any() or np.isnan(y).any():
                    continue
                
                self.X.append(x)
                self.Y.append(y)
                # Use the date of the last timestep
                date_idx = i + self.seq_length - 1
                # Adjust for padding in dates if necessary, or just trace back relative to end
                if date_idx >= len(dates): 
                    # If we padded, we might effectively be at a date index that doesn't exist in original df if we weren't careful.
                    # But here we padded features/targets but 'dates' is still original.
                    # If we are using padded data, we really only care about the last real date.
                    current_date = dates[-1] 
                else: 
                     # If we needed padding, len(features) > len(dates). 
                     # The indices match the END of the arrays. 
                     # A simple mapping: relative to the end.
                     offset_from_end = len(features) - 1 - date_idx
                     original_idx = len(dates) - 1 - offset_from_end
                     current_date = dates[original_idx] if original_idx >= 0 else dates[0]

                self.metadata.append((ticker, current_date))
                
        except Exception as e:
            print(f"  ❌ Error processing {ticker}: {e}")
    
    def _normalize_data(self):
        if self.train:
            X_flat = self.X.reshape(-1, len(self.feature_cols))
            
            self.scaler_params = {
                'min': X_flat.min(axis=0),
                'max': X_flat.max(axis=0),
            }
        
        min_vals = self.scaler_params['min']
        max_vals = self.scaler_params['max']
        
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        
        self.X = (self.X - min_vals) / range_vals
        
        print(f"Data normalized with MinMax scaling (0-1)")
    
    def get_scaler_params(self) -> dict:
        return self.scaler_params
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.Y[idx], dtype=torch.float32)
        return x, y
    
    def get_metadata(self, idx: int) -> Tuple[str, pd.Timestamp]:
        return self.metadata[idx]


def create_dataloaders(
    data_dir: Union[str, Path] = PROCESSED_DIR,
    seq_length: int = 120,
    batch_size: int = 64,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    full_dataset = StockDataset(
        data_dir=data_dir,
        seq_length=seq_length,
        train=True,
    )
    
    scaler_params = full_dataset.get_scaler_params()
    
    n_samples = len(full_dataset)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val
    
    print(f"\nDataset splits:")
    print(f"  - Train: {n_train:,} samples ({train_ratio*100:.0f}%)")
    print(f"  - Val:   {n_val:,} samples ({val_ratio*100:.0f}%)")
    print(f"  - Test:  {n_test:,} samples ({(1-train_ratio-val_ratio)*100:.0f}%)")
    
    torch.manual_seed(seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val, n_test]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader, scaler_params


if __name__ == "__main__":
    print("Testing StockDataset...\n")
    
    if not PROCESSED_DIR.exists() or not list(PROCESSED_DIR.glob("*.csv")):
        print(f"❌ No processed data found in {PROCESSED_DIR}")
        print("Run feature_engineering.py first to create processed data.")
        exit(1)
    
    dataset = StockDataset(seq_length=120)
    
    print(f"\nDataset size: {len(dataset):,} samples")
    
    x, y = dataset[0]
    ticker, date = dataset.get_metadata(0)
    
    print(f"\nSample 0:")
    print(f"  Ticker: {ticker}")
    print(f"  Date: {date}")
    print(f"  X shape: {x.shape}  (seq_length, n_features)")
    print(f"  Y shape: {y.shape}  (Return_3M, Return_1Y, Return_3Y)")
    print(f"  Y values: {y.numpy()}")
    
    print("\n" + "="*50)
    print("Testing DataLoaders...")
    
    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        batch_size=32
    )
    
    batch_x, batch_y = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  X: {batch_x.shape}  (batch, seq_length, n_features)")
    print(f"  Y: {batch_y.shape}  (batch, 3)")
    
    print("\n✅ Dataset test passed!")
