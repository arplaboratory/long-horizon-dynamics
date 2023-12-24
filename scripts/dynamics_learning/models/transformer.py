import torch 
import torch.nn as nn
import math

class Transformer(nn.Module):

    def __init__(self, input_size, encoder_dim, num_heads, history_length, ffn_hidden, num_layers, dropout):
        super(Transformer, self).__init__()

        self.encoder = Encoder(input_size, encoder_dim, num_heads, history_length, ffn_hidden, num_layers, dropout)

    def forward(self, x):

        x_encoded = self.encoder(x)
    
        return x_encoded[:, -1, :]
    

class Encoder(nn.Module):
    def __init__(self, input_size, d_model, num_heads, history_length, ffn_hidden, n_layers, dropout):
        super(Encoder, self).__init__()

        self.embedding = nn.Linear(input_size, d_model)
        self.positional_encoding = self._generate_positional_encoding(history_length, d_model)

        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, ffn_hidden, dropout) for _ in range(n_layers)])

    def _generate_positional_encoding(self, history_length, d_model):
        """
        Generates the positional encoding for the transformer.
        :param history_length: The length of the history
        :param d_model: The number of features
        :return: The positional encoding
        """
        positional_encoding = torch.zeros(1, history_length, d_model)
        position = torch.arange(0, history_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        positional_encoding[0, :, 0::2] = torch.sin(position * div_term)
        positional_encoding[0, :, 1::2] = torch.cos(position * div_term)

        return positional_encoding
    
    def forward(self, x):
        """
        Performs a forward pass through the encoder.
        :param x: The input tensor
        :return: The output tensor
        """
        x = self.embedding(x) + self.positional_encoding.to(x.device)

        for layer in self.layers:
            x = layer(x)
            
        return x
    
class EncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, ffn_hidden, dropout):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = PositionWiseFeedForward(d_model, ffn_hidden)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Performs a forward pass through the encoder layer.
        :param x: The input tensor
        :return: The output tensor
        """

        # Compute self-attention
        _x = x 
        x = self.attention(x, x, x)

        # Add and normalize
        x = self.norm1(_x + self.dropout1(x))

        # Compute position-wise feed-forward
        _x = x
        x = self.ffn(x)

        # Add and normalize
        x = self.norm2(_x + self.dropout2(x))

        return x
    
class ScaledDotProductAttention(nn.Module):

    """
    Computes scaled dot-product attention.
    """
    
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, e=1e-12):
        """
        Performs a forward pass through the scaled dot-product attention.
        :param q: The query tensor
        :param k: The key tensor
        :param v: The value tensor
        :param e: A small value to avoid division by zero
        :return: The output tensor
        """
        # Input is a 4D tensor 
        # q: (batch_size, heads, history_length, d_k)

        batch_size, heads, history_length, d_k = k.size()

        # Compute the dot product Query x Key^T to compute similarity scores
        k_T = k.transpose(2, 3)
        scores = (q @ k_T) / math.sqrt(d_k)

        # Softmax to get attention weights
        scores = self.softmax(scores)

        # Multiply scores with values to get context vectors
        v = scores @ v

        return v, scores
    
class LayerNorm(nn.Module):
    
        def __init__(self, d_model, eps=1e-6):
            super(LayerNorm, self).__init__()
    
            self.eps = eps
    
            self.gamma = nn.Parameter(torch.ones(d_model), requires_grad=True)
            self.beta = nn.Parameter(torch.zeros(d_model), requires_grad=True)
    
        def forward(self, x):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, unbiased=False, keepdim=True)
    
            out = (x - mean) / torch.sqrt(var + self.eps)
            out = self.gamma * out + self.beta
    
            return out
        
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.attention = ScaledDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)  
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        """
        Performs a forward pass through the multi-head attention layer.
        :param q: The query tensor
        :param k: The key tensor
        :param v: The value tensor
        :return: The output tensor
        """

        # Input is a 4D tensor 
        # q: (batch_size, heads, history_length, d_k)

        # Dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # Split into multiple heads
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)

        # Compute scaled dot-product attention
        out, scores = self.attention(q, k, v)

        # Concatenate heads
        out = self.concat_heads(out)

        # Linear layer
        out = self.w_concat(out)

        return out



    def split_heads(self, x):
        """
        Splits the last dimension into (num_heads, d_model/num_heads)
        :param x: [batch_size, history_length, d_model]
        :return: [batch_size, num_heads, history_length, d_model/num_heads]
        """

        batch_size, history_length, d_model = x.size()

        # Compute depth of each head
        depth = d_model // self.num_heads

        # Reshape into (batch_size, history_length, num_heads, depth)
        x = x.view(batch_size, history_length, self.num_heads, depth)

        # Transpose to (batch_size, num_heads, history_length, depth)
        return x.transpose(1, 2)
    
    def concat_heads(self, x):

        batch_size, num_heads, history_length, depth = x.size()

        # Transpose to (batch_size, history_length, num_heads, depth)
        x = x.transpose(1, 2).contiguous()

        # Reshape to (batch_size, history_length, d_model)
        x = x.view(batch_size, history_length, num_heads * depth)

        return x
    
class PositionWiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.relu(self.linear_1(x)))
        x = self.linear_2(x)

        return x
    

if __name__=='__main__':
    input_size = 19 #number of features
    output_dim = 256
    num_heads = 8
    history_length = 10
    ffn_hidden = 512
    num_layers = 2
    dropout = 0.2

    model = Transformer(input_size, output_dim, num_heads, history_length, ffn_hidden, num_layers, dropout)

    x = torch.randn(32, 10, 19)

    y = model(x)

    print(y.shape)
