import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class HybridForecaster(nn.Module):
    def __init__(
        self,
        input_size: int = 5,
        seq_length: int = 120,
        cnn_channels: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.seq_length = seq_length
        self.lstm_hidden = lstm_hidden
        
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=cnn_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=cnn_channels,
                out_channels=cnn_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.bilstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        
        bilstm_output_size = lstm_hidden * 2
        
        self.pos_encoder = PositionalEncoding(
            d_model=bilstm_output_size,
            max_len=seq_length,
            dropout=dropout
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=bilstm_output_size,
            nhead=transformer_heads,
            dim_feedforward=bilstm_output_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
        )
        
        self.feature_pool = nn.AdaptiveAvgPool1d(1)
        
        self.head_3m = nn.Sequential(
            nn.Linear(bilstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        
        self.head_1y = nn.Sequential(
            nn.Linear(bilstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        
        self.head_3y = nn.Sequential(
            nn.Linear(bilstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        
        x = x.permute(0, 2, 1)
        
        x, _ = self.bilstm(x)
        
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        x = x.permute(0, 2, 1)
        x = self.feature_pool(x)
        x = x.squeeze(-1)
        
        pred_3m = self.head_3m(x)
        pred_1y = self.head_1y(x)
        pred_3y = self.head_3y(x)
        
        return pred_3m, pred_1y, pred_3y


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing HybridForecaster...")
    
    batch_size = 32
    seq_length = 120
    input_size = 5
    
    model = HybridForecaster(
        input_size=input_size,
        seq_length=seq_length,
    )
    
    print(f"\nModel Architecture:")
    print(model)
    
    print(f"\nTrainable Parameters: {count_parameters(model):,}")
    
    x = torch.randn(batch_size, seq_length, input_size)
    
    model.eval()
    with torch.no_grad():
        pred_3m, pred_1y, pred_3y = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shapes:")
    print(f"  - pred_3m: {pred_3m.shape}")
    print(f"  - pred_1y: {pred_1y.shape}")
    print(f"  - pred_3y: {pred_3y.shape}")
    
    print("\n✅ Model test passed!")
