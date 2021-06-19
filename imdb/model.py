import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import DPCNN

def load_pretrain(bin_file, freeze=True):
    import gensim
    model = gensim.models.KeyedVectors.load_word2vec_format(bin_file, binary=True)
    weights = torch.FloatTensor(model.vectors)
    zero = torch.zeros((1,weights.size()[-1]))
    weights = torch.cat((zero, weights), 0)
    embedding = nn.Embedding.from_pretrained(weights, freeze=freeze, padding_idx=0)
    return embedding

class TextDPCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, output_size, add_norm=False):
        super(TextDPCNN, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size, padding_idx=None)
        self.dpcnn = DPCNN(embed_size, hidden_size, num_layers, add_norm)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        emb_out = self.embeddings(inputs)
        x = self.dpcnn(emb_out)
        x = self.bn(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_channels, kernel_sizes, output_size, dropout=0.5, spatial_dropout=False):
        super(TextCNN, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size, padding_idx=None)
        self.conv_list = []
        for kernel in kernel_sizes:
            conv = nn.Sequential(
                nn.Conv1d(embed_size, num_channels, kernel_size=kernel, padding=1), # (seq_len+2*padding-kernel_num)/stride+1
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
                )
            self.conv_list.append(conv)
        self.conv_list = nn.ModuleList(self.conv_list)
        hidden_size = num_channels * len(kernel_sizes)
        self.fc = nn.Linear(hidden_size, output_size)
        #self.fc = nn.Sequential(
        #        nn.Linear(num_directions*hidden_size, 64),
        #        nn.Dropout(0.5),
        #        nn.ReLU(),
        #        nn.Linear(64, output_size)
        #        )
        self.spatial_dropout = spatial_dropout
        self.dropout2d =nn.Dropout2d(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        emb_out = self.embeddings(inputs).permute(0,2,1) # (batch_size, embed_size, seq_len)
        if self.spatial_dropout:
            conv_out = [self.dropout2d(conv(emb_out)).squeeze(2) for conv in self.conv_list] # (batch_size, num_channels, 1)
        else:
            conv_out = [conv(emb_out).squeeze(2) for conv in self.conv_list] # (batch_size, num_channels, 1)
        concat_out = torch.cat(conv_out, dim=1) # (batch_size, num_channels*len(kernel_sizes))
        concat_out = self.dropout(concat_out)
        x = self.fc(concat_out)
        x = F.log_softmax(x, dim=1)
        return x

class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, output_size, bi_flag=True, dropout=0):
        super(TextLSTM, self).__init__()
        num_directions = 1
        if bi_flag:
            num_directions = 2
        self.embeddings = nn.Embedding(vocab_size, embed_size, padding_idx=None) # not set padding_idx=0
        if num_layers <= 1:
            self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=1, bidirectional=bi_flag)
        else:
            self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, bidirectional=bi_flag, dropout=0.2)
        self.fc = nn.Linear(num_directions*hidden_size, output_size)

    def forward(self, inputs):
        x = self.embeddings(inputs).permute(1,0,2) # (seq_len, batch_size, embed_size)
        lstm_out,lstm_h = self.lstm(x) # (seq_len, batch_size, hidden_size)
        x = lstm_out[-1,:,:] #get last output , adding dropout gets worse
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

class TextTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, output_size, dropout=0.1):
        super(TextTransformer, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embed_size, padding_idx=None) # not set padding_idx=0
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=8, dim_feedforward=hidden_size, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=None)
        self.fc = nn.Linear(embed_size, output_size)

    def forward(self, inputs):
        x = self.embeddings(inputs) # (batch_size, seq_len, embed_size)
        x = self.transformer_encoder(x) # (batch_size, seq_len, embed_size)
        x = self.fc(x.mean(1))
        x = F.log_softmax(x, dim=1)
        return x
