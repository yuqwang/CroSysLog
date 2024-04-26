import torch
import torch.nn as nn


class GRUAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, sequence_length):
        super(GRUAutoencoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length

        # 编码器
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)

        # 解码器
        self.decoder = nn.GRU(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        # 编码
        _, hidden = self.encoder(x)

        # 重复最后的隐藏状态，准备作为解码器的输入
        repeated_hidden = hidden.repeat(self.sequence_length, 1, 1).permute(1, 0, 2)

        # 解码
        decoded, _ = self.decoder(repeated_hidden)

        return decoded