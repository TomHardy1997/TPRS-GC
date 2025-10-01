import torch
import torch.nn as nn


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, max_positions: int = 1000) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(max_positions, embedding_dim)
    
    def forward(self, input, mask):
        """
        Args:
            input: Input tensor of shape (batch_size, sequence_length, feature_dim)
            mask: Mask tensor of shape (batch_size, sequence_length)
        
        Returns:
            Position embeddings of shape (batch_size, sequence_length, embedding_dim)
        """
        batch_size, sequence_length, _ = input.shape
        
        # Generate positions based on mask
        positions = self._make_positions(mask)
        return self.embedding(positions)
    
    def _make_positions(self, mask):
        """
        Generate position indices using mask.
        Valid positions (mask=1) get incremental position indices,
        padded positions (mask=0) remain 0.
        """
        return torch.cumsum(mask.long(), dim=1) * mask.long()
