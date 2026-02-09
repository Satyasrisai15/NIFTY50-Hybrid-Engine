import os
import sys
from pathlib import Path

from google.colab import drive
drive.mount('/content/drive')

DRIVE_PATH = Path("/content/drive/MyDrive/Nifty50-predictor")
sys.path.append(str(DRIVE_PATH / "src"))

from dataset import StockDataset
from model import HybridForecaster

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROCESSED_DIR = DRIVE_PATH / "data" / "processed"
MODELS_DIR = DRIVE_PATH / "models"

SEQ_LENGTH = 120
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 50

device = torch.device('cuda')
print(f"Using device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

MODELS_DIR.mkdir(parents=True, exist_ok=True)

csv_files = sorted(PROCESSED_DIR.glob("*.csv"))
tickers = [f.stem for f in csv_files]

print(f"\nFound {len(tickers)} stocks to train:")
print(", ".join(tickers))


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for X, Y in dataloader:
        X = X.to(device)
        Y = Y.to(device)
        
        optimizer.zero_grad()
        
        pred_3m, pred_1y, pred_3y = model(X)
        
        loss_3m = criterion(pred_3m.squeeze(), Y[:, 0])
        loss_1y = criterion(pred_1y.squeeze(), Y[:, 1])
        loss_3y = criterion(pred_3y.squeeze(), Y[:, 2])
        
        loss = loss_3m + loss_1y + loss_3y
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def train_stock(ticker, processed_dir, models_dir, device):
    print(f"\n{'='*60}")
    print(f"Training model for: {ticker}")
    print(f"{'='*60}")
    
    try:
        dataset = StockDataset(
            data_dir=processed_dir,
            seq_length=SEQ_LENGTH,
            tickers=[ticker],
            train=True,
        )
    except Exception as e:
        print(f"❌ Error loading {ticker}: {e}")
        return None
    
    if len(dataset) == 0:
        print(f"❌ No samples for {ticker}, skipping...")
        return None
    
    print(f"Dataset size: {len(dataset)} samples")
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    
    sample_x, _ = dataset[0]
    input_size = sample_x.shape[1]
    
    model = HybridForecaster(
        input_size=input_size,
        seq_length=SEQ_LENGTH,
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    best_loss = float('inf')
    
    for epoch in tqdm(range(EPOCHS), desc=f"Training {ticker}"):
        avg_loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.6f}")
    
    model_path = models_dir / f"{ticker}_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_params': dataset.get_scaler_params(),
        'best_loss': best_loss,
        'epochs': EPOCHS,
        'ticker': ticker,
    }, model_path)
    
    print(f"✅ Model saved: {model_path}")
    print(f"   Best loss: {best_loss:.6f}")
    
    return best_loss


print(f"\n{'#'*60}")
print(f"STARTING TRAINING FOR ALL {len(tickers)} STOCKS")
print(f"{'#'*60}")

results = {}

for i, ticker in enumerate(tickers):
    print(f"\n[{i+1}/{len(tickers)}] Processing {ticker}...")
    
    loss = train_stock(ticker, PROCESSED_DIR, MODELS_DIR, device)
    
    if loss is not None:
        results[ticker] = loss
    
    torch.cuda.empty_cache()

print(f"\n{'#'*60}")
print(f"TRAINING COMPLETE!")
print(f"{'#'*60}")

print(f"\nSuccessfully trained: {len(results)}/{len(tickers)} models")

if results:
    print(f"\nResults summary:")
    print(f"{'-'*40}")
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    for ticker, loss in sorted_results:
        print(f"  {ticker:15} | Loss: {loss:.6f}")
    
    print(f"\nBest performing: {sorted_results[0][0]} (loss: {sorted_results[0][1]:.6f})")
    print(f"Worst performing: {sorted_results[-1][0]} (loss: {sorted_results[-1][1]:.6f})")

print(f"\nModels saved to: {MODELS_DIR}")
print(f"Total models: {len(list(MODELS_DIR.glob('*.pt')))}")

