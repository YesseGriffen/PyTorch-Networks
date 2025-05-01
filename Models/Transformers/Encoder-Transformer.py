import math
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader



#---------------------------------------------
# Sinusoidal Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        seq_len = x.size(0)
        x = x + self.pe[:seq_len]
        return x
#---------------------------------------------
# Encoder-Only Transformer. *In this use-case we assume our forecast_size = 1, if we need to forecast more than 1, we would benefit from an autoregressive-decoder part, which this doesn't have.*
class Transformer(nn.Module):
    def __init__(self, model_params: list):
        super().__init__()
        input_dim, dim_feedforward, output_dim, d_model, nhead, num_layers, dropout = model_params

        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            activation='relu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, output_dim)

    #---------------------------------------------
    # x = (batch_size, sequence_length, input_dim)
    def forward(self, x: torch.Tensor):
        x = self.input_linear(x)
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)

        enc_out = self.encoder(x)

        last_step = enc_out[-1]
        out = self.fc_out(last_step)

        return out

#---------------------------------------------
# Model OverHead, meaning functions directly related to our current model, EX: fit, forecast, evalute, etc.
class ModelOH:
    def __init__(self, model_params: list, lr: int):
        self.device = torch.device("cuda")
        self.model = Transformer(model_params).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.HuberLoss()
        
    #---------------------------------------------
    # Main training using gradient descent to adjust weights (Supervised, based on Y loss)
    def fit(self, loader: DataLoader, epochs: int):
        for epoch in range(1, epochs+1):
            self.model.train()
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(xb)
                loss = self.criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item() * xb.size(0)

            

            if epoch == epochs or epoch % 30 == 0:
                print(f"□□ Epoch {epoch}/{epochs} ▪︎ Loss: {total_loss/len(loader.dataset):.4f}")

    #---------------------------------------------
    # Made for smaller updates where learning rate is reduced before training *And returned to normal after*
    def update_fit(self, loader: DataLoader, epochs: int, lr_factor: int = 0.1):
        orig_lrs = [g['lr'] for g in self.optimizer.param_groups]
        for g in self.optimizer.param_groups:
            g['lr'] *= lr_factor
        
        self.fit(loader, epochs)
        

        for g, lr in zip(self.optimizer.param_groups, orig_lrs):
            g['lr'] = lr
    #---------------------------------------------
    # Evaluate Loss on DataLoader
    def evaluate(self, loader: DataLoader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                preds = self.model(xb)
                total_loss += self.criterion(preds, yb).item() * xb.size(0)
            return total_loss / len(loader.dataset)
    #---------------------------------------------
    # Forecast prediction *Made for forecast_size = 1*
    def forecast(self, x: torch.Tensor):
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            y = self.model(x)
            return y