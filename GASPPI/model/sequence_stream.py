from torch import nn
from .base import InteractionBlock

class MambaSequenceEncoder(nn.Module):
    """
    Encodes a sequence of amino acid IDs using Mamba blocks.

    This module takes a sequence of integer IDs, embeds them into a
    high-dimensional space, and then processes them through a series of
    Mamba blocks to capture long-range dependencies and contextual
    information within the sequence.
    """
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 heads: int,
                 dropout: float,
                 mamba_d_state: int,
                 mamba_d_conv: int,
                 mamba_expand: int):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Project embedding to hidden dimension if they are different
        if embedding_dim != hidden_dim:
            self.embedding_proj = nn.Linear(embedding_dim, hidden_dim)
        else:
            self.embedding_proj = nn.Identity()

        self.encoder_blocks = nn.ModuleList([
            InteractionBlock(
                hidden_dim=hidden_dim,
                heads=heads,  # heads are not used in Mamba, but kept for consistency
                dropout=dropout,
                use_gnn=False,
                use_mamba=True,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, sequence_ids):
        """
        Args:
            sequence_ids (torch.Tensor): A tensor of shape (batch_size, seq_len)
                                         containing amino acid integer IDs.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, seq_len, hidden_dim)
                          representing the deep embeddings for the sequence.
        """
        # (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        x = self.embedding(sequence_ids)
        
        # (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, hidden_dim)
        x = self.embedding_proj(x)

        for block in self.encoder_blocks:
            # InteractionBlock expects shape (num_nodes, hidden_dim) or (batch, num_nodes, hidden_dim)
            # Mamba layer inside handles batch dimension correctly
            x = block(x)

        x = self.norm(x)
        return x 