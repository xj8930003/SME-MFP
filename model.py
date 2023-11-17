import torch
import torch.nn as nn
from torch.nn import init
import numpy as np

class AFT_FULL(nn.Module):

    def __init__(self, d_model, n=49, simple=False):

        super(AFT_FULL, self).__init__()
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        if (simple):
            self.position_biases = torch.zeros((n, n))
        else:
            self.position_biases = nn.Parameter(torch.ones((n, n)))
        self.d_model = d_model
        self.n = n
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):

        bs, n, dim = input.shape

        q = self.fc_q(input)  # bs,n,dim
        k = self.fc_k(input).view(1, bs, n, dim)  # 1,bs,n,dim
        v = self.fc_v(input).view(1, bs, n, dim)  # 1,bs,n,dim

        numerator = torch.sum(
            torch.exp(k + self.position_biases.view(n, 1, -1, 1)) * v,
            dim=2)  # n,bs,dim
        denominator = torch.sum(
            torch.exp(k + self.position_biases.view(n, 1, -1, 1)),
            dim=2)  # n,bs,dim

        out = (numerator / denominator)  # n,bs,dim
        out = self.sigmoid(q) * (out.permute(1, 0, 2))  # bs,n,dim

        return out

class Mymodel(nn.Module):
    def __init__(self,
                 vocab_size,
                 Embed_D,
                 pool_size,
                 dropout,
                 output_length,
                 convDim,
                 hidden_size,
                 Attention_Dim,
                 Max_length,
                 N_head,
                 embedding
                 ):
        super(Mymodel, self).__init__()

        self.Max_length = Max_length
        self.Embed1 = nn.Embedding.from_pretrained(torch.FloatTensor(embedding[0]))
        self.Embed2 = nn.Embedding.from_pretrained(torch.FloatTensor(embedding[1]))
        self.Dropout = nn.Dropout(dropout)  # 强相关性能不能丢弃
        self.conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=Embed_D,
                    out_channels=convDim,
                    kernel_size=h,
                    padding=0),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=pool_size, stride=1, padding=1))
            for h in [2, 3, 8]
        ])

        self.lstm = nn.LSTM(243, hidden_size, bidirectional=True, batch_first=True)
        self.MultiHeadAttention = AFT_FULL(d_model=200, n=50)

        self.Dense = nn.Sequential(
            nn.Linear(10000, 128),
            nn.ReLU(),
            nn.Linear(128, output_length),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.Embed1(x).permute(0,2,1)
        x2 = self.Embed2(x).permute(0,2,1)
        x0 = torch.cat([x1,x2],dim=1)

        out = [conv(x0) for conv in self.conv]

        # 这里写成自动化
        fun0 = torch.nn.ZeroPad2d(padding=(0, self.Max_length - out[0].shape[2], 0, 0, 0, 0))
        fun1 = torch.nn.ZeroPad2d(padding=(0, self.Max_length - out[1].shape[2], 0, 0, 0, 0))
        fun2 = torch.nn.ZeroPad2d(padding=(0, self.Max_length - out[2].shape[2], 0, 0, 0, 0))

        x1 = torch.cat([fun0(out[0]), fun1(out[1]), fun2(out[2])], dim=1)

        x1 = self.Dropout(x1)
        merge = torch.cat([x0,x1],dim=1)

        x, _ = self.lstm(merge.permute(0, 2, 1))

        x = self.MultiHeadAttention(x)

        x = self.Dense(x.flatten(1))

        return x
