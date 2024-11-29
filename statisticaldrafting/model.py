import torch
import torch.nn as nn
import torch.nn.functional as F

class DraftMLP(nn.Module):
    def __init__(self,
                 cardnames,
                 hidden_dims,
                 dropout=0.1):
        """
        Args:
            cardnames (List[str]): Names of cards in the set. 
            hidden_dims (list): List of integers specifying hidden layer sizes (must be equal).
            dropout (float): Dropout rate for regularization.
        """
        super(DraftMLP, self).__init__()
        
        # Customize to given set. 
        self.cardnames = cardnames

        # Input layer
        self.input_layer = nn.Linear(len(cardnames), hidden_dims[0])
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_dims[i], hidden_dims[i+1]) for i in range(len(hidden_dims) - 1)
        )
        
        # Projection layers for residuals if dimensions mismatch
        self.projections = nn.ModuleList(
            nn.Linear(hidden_dims[i], hidden_dims[i+1]) if hidden_dims[i] != hidden_dims[i+1] else nn.Identity()
            for i in range(len(hidden_dims) - 1)
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], len(cardnames))
        
        # Normalization and regularization
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for dim in hidden_dims)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, pack):
        # Input layer
        x = self.input_layer(x)
        x = F.gelu(x)  # Activation function
        x = self.dropout(x)
        
        # Hidden layers with residual connections
        for layer, norm, proj in zip(self.hidden_layers, self.norms, self.projections):
            residual = x
            x = layer(x)
            x = F.gelu(x)
            x = self.dropout(x)
            x = norm(x + proj(residual))  # Residual connection + LayerNorm
            
        # Output layer
        x = self.output_layer(x)
        x = x * pack
        return x