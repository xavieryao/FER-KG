import torch.nn.functional as F
from torch import nn
from model import SavableModel


class EmbRegressionModel(SavableModel):
    def __init__(self, embed_dim, num_contexts, context_length, num_layers):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_contexts = num_contexts
        self.context_length = context_length
        self.num_layers = num_layers

        self.context_encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=2, dim_feedforward=128)
        self.context_encoder = nn.TransforerEncoder(self.context_encoder_layer, num_layers)

        self.context_pool = nn.MaxPool1d(context_length)

        self.shot_encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=2, dim_feedforward=128)
        self.shot_encoder = nn.TransforerEncoder(self.context_encoder_layer, num_layers)

        self.output = nn.Linear(embed_dim * num_contexts, embed_dim)
