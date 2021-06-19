import torch
import torch.nn as nn

class DPCNNResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, padding, downsample, add_norm=False):
        super(DPCNNResidualBlock, self).__init__()
        if add_norm:
            self.residual = nn.Sequential(
                nn.BatchNorm1d(num_features=channels),
                nn.ReLU(),
                nn.Conv1d(
                    in_channels=channels, out_channels=channels,
                    kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(num_features=channels),
                nn.ReLU(),
                nn.Conv1d(
                    in_channels=channels, out_channels=channels,
                    kernel_size=kernel_size, padding=padding)
            )
        else:
            self.residual = nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(
                    in_channels=channels, out_channels=channels,
                    kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
                nn.Conv1d(
                    in_channels=channels, out_channels=channels,
                    kernel_size=kernel_size, padding=padding)
            )
        self.pool = nn.MaxPool1d(2)
        self.downsample = downsample

    def forward(self, x):
        output = self.residual(x)
        output = x + output
        if self.downsample:
            output = self.pool(output)
        return output


class DPCNN(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers=7, add_norm=False):
        super(DPCNN, self).__init__()
        self.cnn = nn.Conv1d(
            in_channels=embed_size, out_channels=hidden_size,
            kernel_size=1, padding=0
        )
        self.residual_layer = self._make_layer(num_layers, hidden_size,
                                               kernel_size=3, padding=1,
                                               downsample=True, add_norm=add_norm)
        self.globalpool = nn.AdaptiveAvgPool2d((None, 1))

    def _make_layer(self, num_layers, channels,
                    kernel_size, padding, downsample, add_norm):
        layers = []
        for _ in range(num_layers-1):
            layers.append(DPCNNResidualBlock(channels, kernel_size,
                                             padding, downsample, add_norm=add_norm))
        layers.append(DPCNNResidualBlock(channels, kernel_size,
                                         padding, downsample=False, add_norm=add_norm))
        return nn.Sequential(*layers)

    def forward(self, x):
        embeds = x.permute(0, 2, 1)
        output = self.cnn(embeds)
        output = self.residual_layer(output)
        #print("size:",output.size()) #size: torch.Size([32, 256, 2])
        output = self.globalpool(output).squeeze(dim=-1)
        return output
