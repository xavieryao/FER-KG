import torch.nn.functional as F
from torch import nn
import torch
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

    def forward(self, x: torch.Tensor):
        # Input: B x C x L x N
        # Output: N x L
        batch_size, num_contexts, context_length, num_embeddings = x.shape

        # context encoding
        context_encodings = []
        for c in range(num_contexts):
            ctx = x[:, c].transpose(0, 1)  # L x B x N
            ctx: torch.Tensor = self.context_encoder(ctx)  # L x B x N
            ctx = ctx.permute(1, 2, 0)  # B x N x L
            ctx = self.context_pool(ctx).squeeze()  # B x N x 1
            context_encodings.append(ctx)
        context_encodings = torch.stack(context_encodings)  # C x B x N

        shot_encoding = self.shot_encoder(context_encodings)  # C x B x N
        shot_encoding = shot_encoding.transpose(0, 1)  # B x C x N
        shot_encoding = shot_encoding.reshape(batch_size, -1)  # B x N

        output = self.output(shot_encoding)
        return F.normalize(output, p=2, dim=-1)