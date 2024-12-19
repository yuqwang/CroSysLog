import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class Learner(nn.Module):

    def __init__(self, config):
        """
        :param config: network config file, type:list of (string, list)
        """
        super(Learner, self).__init__()

        self.config = config

        for i, (name, param) in enumerate(self.config):
            if name == 'AEGRU':
                """
                  :param config: [input_size, hidden_size, sequence_length]
                        """
                self.input_size = param[0]
                self.hidden_size = param[1]

                # Encoder
                self.encoder = nn.GRU(self.input_size, self.hidden_size, batch_first=True)

                # Decoder
                self.decoder = nn.GRU(self.hidden_size, self.input_size, batch_first=True)

            elif name == 'LSTM':
                self.input_size = param[0]
                self.hidden_size = param[1]
                self.n_layers = param[2]
                self.output_size = param[3]
                self.dropout_rate = param[4]

                self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.n_layers,
                                    dropout=self.dropout_rate, batch_first=True)
                self.linear = nn.Linear(self.hidden_size, self.output_size)
                self.hidden2 = None

            elif name == 'transformer_encoder':
                #print("transformer_encoder is initialized")

                """
                :param config: [nhead, num_encoder_layers, num_decoder_layers, d_model, dim_feedforward]
                """
                self.d_model = param[0]
                self.nhead = param[1]
                self.num_encoder_layers = param[2]
                self.dim_feedforward = param[3]
                self.output_size = param[4]
                self.dropout_rate = param[5]

                # Define transformer encoder layer
                encoder_layers = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                            nhead=self.nhead,
                                                            dim_feedforward=self.dim_feedforward,
                                                            dropout=self.dropout_rate,
                                                            batch_first=True)

                # Define transformer encoder
                self.transformer_encoder = nn.TransformerEncoder(encoder_layers,
                                                                 num_layers=self.num_encoder_layers)

                self.linear = nn.Linear(self.d_model, self.output_size)

            else:
                raise NotImplementedError

    def forward(self, x):
        """
                This function can be called by forward & finetunning from MAML.
                :param x: [setsz, embedding_length]
                :return: x, loss, likelihood, kld
                """
        for name, param in self.config:
            if name == 'AEGRU':
                _, z = self.encoder(x)

                repeated_z = z.repeat(self.sequence_length, 1, 1).permute(1, 0, 2)

                recon_x, _ = self.decoder(repeated_z)
                output = recon_x

            elif name == 'transformer_encoder':
                output = self.transformer_encoder(x)

                output = self.linear(output)

            elif name == 'LSTM':
                setsz = x.size(0)

                if self.hidden2 is None or self.hidden2[0].size(1) != setsz:
                    self.init_hidden(setsz, x.is_cuda, mode="lstm")

                output, hidden = self.lstm(x, self.hidden2)

                # Instead of taking the last timestep, process all timesteps
                output = output.contiguous().view(-1, self.hidden_size)

                # Pass the output of each timestep through the linear layer
                output = self.linear(output)

                # Reshape back to [batch_size, sequence_length, output_size]
                output = output.view(setsz, -1, self.output_size)

        return output

    def updated_forward(self, x, fast_weights):
        for p, new_p in zip(self.parameters(), fast_weights):
            p.data = new_p.data
        output = self.forward(x)
        return output

    def update_para_forward(self, x, w):
        weight = w[:-1]
        bias = w[-1]

        self.decoder[0].weight.data = weight
        self.decoder[0].bias.data = bias

        output = self.forward(x)
        return output

    def update_paras(self, w):
        weight = w[:-1]
        bias = w[-1]
        fast_parameters = [weight] + [bias]
        self.decoder[0].weight.data = weight
        self.decoder[0].bias.data = bias

    def init_hidden(self, input_setsz, gpu = True, mode ="rnn"):
        weight = next(self.parameters()).data
        if mode == 'rnn':
            if (gpu):
                self.hidden1 = weight.new(self.n_layers, input_setsz, self.hidden_size).zero_().cuda()
            else:
                self.hidden1 =weight.new(self.n_layers, input_setsz, self.hidden_size).zero_()

        elif mode == 'lstm':
            if (gpu):
                self.hidden2 = (weight.new(self.n_layers, input_setsz, self.hidden_size).zero_().cuda(),
                                weight.new(self.n_layers, input_setsz, self.hidden_size).zero_().cuda())
            else:
                self.hidden2 = (weight.new(self.n_layers, input_setsz, self.hidden_size).zero_(),
                                weight.new(self.n_layers, input_setsz, self.hidden_size).zero_())

