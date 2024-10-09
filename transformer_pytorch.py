import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Implements Multi-Head Attention mechanism as described in "Attention is All You Need".

    Attributes:
        d_k (int): Dimension of the key/query vectors.
        n_heads (int): Number of attention heads.
        key (nn.Linear): Linear layer to project input to key vectors.
        query (nn.Linear): Linear layer to project input to query vectors.
        value (nn.Linear): Linear layer to project input to value vectors.
        fc (nn.Linear): Final linear layer to project concatenated heads back to model dimension.
    """

    def __init__(self, d_k: int, d_model: int, n_heads: int):
        super(MultiHeadAttention, self).__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads."

        self.d_k = d_k
        self.n_heads = n_heads
        self.d_v = d_k  # Assuming d_v = d_k

        # Define linear layers for key, query, and value
        self.key = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.query = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.value = nn.Linear(d_model, d_k * n_heads, bias=False)

        # Final linear layer after concatenation of all heads
        self.fc = nn.Linear(d_k * n_heads, d_model, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for Multi-Head Attention.

        Args:
            query (torch.Tensor): Input tensor for queries of shape (N, T, D).
            key (torch.Tensor): Input tensor for keys of shape (N, T, D).
            value (torch.Tensor): Input tensor for values of shape (N, T, D).
            mask (Optional[torch.Tensor]): Attention mask of shape (N, T).

        Returns:
            torch.Tensor: Output tensor after attention mechanism of shape (N, T, D).
        """
        N, T, _ = query.size()

        # Linear projections
        Q = self.query(query)  # (N, T, n_heads * d_k)
        K = self.key(key)      # (N, T, n_heads * d_k)
        V = self.value(value)  # (N, T, n_heads * d_k)

        # Reshape and transpose for multi-head attention
        Q = Q.view(N, T, self.n_heads, self.d_k).transpose(1, 2)  # (N, n_heads, T, d_k)
        K = K.view(N, T, self.n_heads, self.d_k).transpose(1, 2)  # (N, n_heads, T, d_k)
        V = V.view(N, T, self.n_heads, self.d_v).transpose(1, 2)  # (N, n_heads, T, d_v)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (N, n_heads, T, T)

        if mask is not None:
            # Expand mask to match attention scores shape
            mask = mask.unsqueeze(1).unsqueeze(2)  # (N, 1, 1, T)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)  # (N, n_heads, T, T)
        attn_weights = F.dropout(attn_weights, p=0.1, training=self.training)  # Optional dropout

        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, V)  # (N, n_heads, T, d_v)

        # Concatenate all heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(N, T, self.d_k * self.n_heads)  # (N, T, n_heads * d_k)

        # Final linear projection
        output = self.fc(attn_output)  # (N, T, d_model)

        return output


class PositionwiseFeedForward(nn.Module):
    """
    Implements the Position-wise Feed-Forward Network.

    Attributes:
        ann (nn.Sequential): Sequential container of linear, activation, and dropout layers.
    """

    def __init__(self, d_model: int, d_ff: int, dropout_prob: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.ann = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout_prob),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the position-wise feed-forward network.

        Args:
            x (torch.Tensor): Input tensor of shape (N, T, D).

        Returns:
            torch.Tensor: Output tensor of shape (N, T, D).
        """
        return self.ann(x)


class TransformerBlock(nn.Module):
    """
    Implements a single Transformer block consisting of Multi-Head Attention and Feed-Forward Network.

    Attributes:
        ln1 (nn.LayerNorm): Layer normalization for the residual connection after attention.
        ln2 (nn.LayerNorm): Layer normalization for the residual connection after feed-forward.
        mha (MultiHeadAttention): Multi-head attention mechanism.
        ff (PositionwiseFeedForward): Position-wise feed-forward network.
        dropout (nn.Dropout): Dropout layer applied after residual connections.
    """

    def __init__(
        self,
        d_k: int,
        d_model: int,
        n_heads: int,
        d_ff: int = 4 * 64,  # Typically 4 times d_model
        dropout_prob: float = 0.1
    ):
        super(TransformerBlock, self).__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_k, d_model, n_heads)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout_prob)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (N, T, D).
            mask (Optional[torch.Tensor]): Attention mask of shape (N, T).

        Returns:
            torch.Tensor: Output tensor of shape (N, T, D).
        """
        # Multi-Head Attention with residual connection
        attn_output = self.mha(x, x, x, mask)  # (N, T, D)
        x = self.ln1(x + self.dropout(attn_output))  # (N, T, D)

        # Feed-Forward Network with residual connection
        ff_output = self.ff(x)  # (N, T, D)
        x = self.ln2(x + self.dropout(ff_output))  # (N, T, D)

        return x


class PositionalEncoding(nn.Module):
    """
    Implements positional encoding as described in "Attention is All You Need".

    Attributes:
        dropout (nn.Dropout): Dropout layer applied after adding positional encoding.
        pe (torch.Tensor): Precomputed positional encodings.
    """

    def __init__(self, d_model: int, max_len: int = 2048, dropout_prob: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_prob)

        # Create positional encodings
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model/2)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        # Register as buffer to avoid updating during training
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (N, T, D).

        Returns:
            torch.Tensor: Output tensor with positional encoding added, shape (N, T, D).
        """
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)


class Encoder(nn.Module):
    """
    Implements the Transformer Encoder composed of multiple Transformer blocks.

    Attributes:
        embedding (nn.Embedding): Embedding layer to convert token indices to embeddings.
        pos_encoding (PositionalEncoding): Positional encoding module.
        transformer_blocks (nn.Sequential): Sequential container of Transformer blocks.
        ln (nn.LayerNorm): Final layer normalization.
        fc (nn.Linear): Fully connected layer for output projection.
    """

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_k: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        n_classes: int,
        dropout_prob: float = 0.1,
        d_ff: Optional[int] = None
    ):
        """
        Initializes the Transformer Encoder.

        Args:
            vocab_size (int): Size of the vocabulary.
            max_len (int): Maximum sequence length.
            d_k (int): Dimension of the key/query vectors.
            d_model (int): Dimension of the model.
            n_heads (int): Number of attention heads.
            n_layers (int): Number of Transformer blocks.
            n_classes (int): Number of output classes.
            dropout_prob (float, optional): Dropout probability. Defaults to 0.1.
            d_ff (Optional[int], optional): Dimension of the feed-forward network. Defaults to 4 * d_model.
        """
        super(Encoder, self).__init__()

        if d_ff is None:
            d_ff = 4 * d_model  # Common choice

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout_prob)

        # Create Transformer blocks
        transformer_blocks = [
            TransformerBlock(d_k, d_model, n_heads, d_ff, dropout_prob)
            for _ in range(n_layers)
        ]
        self.transformer_blocks = nn.ModuleList(transformer_blocks)

        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, n_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initializes weights of the model using Xavier uniform initialization for linear layers.
        """
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for the Transformer Encoder.

        Args:
            x (torch.Tensor): Input tensor of token indices, shape (N, T).
            mask (Optional[torch.Tensor], optional): Attention mask, shape (N, T). Defaults to None.

        Returns:
            torch.Tensor: Output logits, shape (N, n_classes).
        """
        x = self.embedding(x)  # (N, T, D)
        x = self.pos_encoding(x)  # (N, T, D)
        for block in self.transformer_blocks:
            x = block(x, mask) # (N, T, D)
        x = x[:, 0, :]  # (N, D) - Selecting the first token's representation
        x = self.ln(x)  # (N, D)
        logits = self.fc(x)  # (N, n_classes)
        return logits


def build_transformer_encoder(
    vocab_size: int,
    max_len: int,
    d_k: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    n_classes: int,
    dropout_prob: float = 0.1,
    d_ff: Optional[int] = None
) -> Encoder:
    """
    Factory function to build a Transformer Encoder.

    Args:
        vocab_size (int): Size of the vocabulary.
        max_len (int): Maximum sequence length.
        d_k (int): Dimension of the key/query vectors.
        d_model (int): Dimension of the model.
        n_heads (int): Number of attention heads.
        n_layers (int): Number of Transformer blocks.
        n_classes (int): Number of output classes.
        dropout_prob (float, optional): Dropout probability. Defaults to 0.1.
        d_ff (Optional[int], optional): Dimension of the feed-forward network. Defaults to 4 * d_model.

    Returns:
        Encoder: Configured Transformer Encoder model.
    """
    return Encoder(
        vocab_size=vocab_size,
        max_len=max_len,
        d_k=d_k,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        n_classes=n_classes,
        dropout_prob=dropout_prob,
        d_ff=d_ff
    )


if __name__ == "__main__":
    # Example instantiation of the Encoder model
    model = build_transformer_encoder(
        vocab_size=20_000,
        max_len=1024,
        d_k=16,
        d_model=64,
        n_heads=4,
        n_layers=2,
        n_classes=5,
        dropout_prob=0.1,
    )

    # Print model architecture
    print(model)