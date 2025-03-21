import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim=1, d_model=128, nhead=4, num_layers=2, num_classes=2):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)  # Add input dimension
        x = self.embed(x)  # Project to d_model
        x = self.transformer(x)  # Pass through Transformer layers
        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)

def get_model(device):
    model = TransformerModel().to(device)
    return model
