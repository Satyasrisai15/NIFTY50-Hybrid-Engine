import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import StockDataset, PROCESSED_DIR
from src.models.hybrid import HybridForecaster
from src.training.trainer import train_one_epoch


def main():
    print("=" * 60)
    print("NIFTY50 Stock Predictor - Local Training Test")
    print("=" * 60)
    
    TICKER = "TMPV"
    SEQ_LENGTH = 120
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    ticker_file = PROCESSED_DIR / f"{TICKER}.csv"
    if not ticker_file.exists():
        print(f"\n❌ Error: {ticker_file} not found!")
        print("Please run src/data/download.py and src/data/feature_engineering.py first.")
        return
    
    print(f"\nLoading dataset for {TICKER}...")
    try:
        dataset = StockDataset(
            seq_length=SEQ_LENGTH,
            tickers=[TICKER],
            train=True,
        )
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    if len(dataset) == 0:
        print("❌ No samples in dataset. Check your processed data.")
        return
    
    print(f"Dataset size: {len(dataset)} samples")
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    print(f"Batches per epoch: {len(dataloader)}")
    
    sample_x, sample_y = dataset[0]
    input_size = sample_x.shape[1]
    print(f"Input shape: ({SEQ_LENGTH}, {input_size})")
    print(f"Target shape: {sample_y.shape}")
    
    print("\nInitializing model...")
    model = HybridForecaster(
        input_size=input_size,
        seq_length=SEQ_LENGTH,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\n{'='*60}")
    print(f"Training for {EPOCHS} epoch(s)...")
    print(f"{'='*60}")
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 40)
        
        avg_loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        
        print(f"\n📊 Epoch {epoch + 1} Average Loss: {avg_loss:.6f}")
    
    print(f"\n{'='*60}")
    print("Testing inference...")
    print(f"{'='*60}")
    
    model.eval()
    with torch.no_grad():
        test_x, test_y = dataset[0]
        test_x = test_x.unsqueeze(0).to(device)
        
        pred_3m, pred_1y, pred_3y = model(test_x)
        
        print(f"\nSample prediction:")
        print(f"  Predicted 3M return:  {pred_3m.item():.4f}")
        print(f"  Predicted 1Y return:  {pred_1y.item():.4f}")
        print(f"  Predicted 3Y return:  {pred_3y.item():.4f}")
        print(f"\nActual values:")
        print(f"  Actual 3M return:     {test_y[0].item():.4f}")
        print(f"  Actual 1Y return:     {test_y[1].item():.4f}")
        print(f"  Actual 3Y return:     {test_y[2].item():.4f}")
    
    print(f"\n{'='*60}")
    print("✅ Training test completed successfully!")
    print("   No bugs detected in the pipeline.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
