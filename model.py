import torch
import torch.nn as nn
from torch.nn import init
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, output_dim, num_head, Input_dim):
        super(MultiHeadAttention, self).__init__()

        self.output_dim = output_dim
        self.num_head = num_head

        params1 = torch.ones((self.num_head, 3, Input_dim, self.output_dim), requires_grad=True)
        self.w = nn.Parameter(params1)

        params2 = torch.ones((self.num_head * self.output_dim, self.output_dim), requires_grad=True)

        self.wo = nn.Parameter(params2)

        self.softmax = nn.Softmax(-1)

        self.init_weights()

    def forward(self, x):
        q = torch.matmul(x, self.w[0, 0])
        k = torch.matmul(x, self.w[0, 1])
        v = torch.matmul(x, self.w[0, 2])

        e = torch.matmul(q, k.transpose(1, 2))
        e = e / (self.output_dim ** 0.5)
        e = self.softmax(e)

        outputs = torch.matmul(e, v)

        for i in range(1, self.w.shape[0]):
            q = torch.matmul(x, self.w[i, 0])
            k = torch.matmul(x, self.w[i, 1])
            v = torch.matmul(x, self.w[i, 2])

            e = torch.matmul(q, k.transpose(1, 2))
            e = e / (self.output_dim ** 0.5)
            e = self.softmax(e)
            o = torch.matmul(e, v)
            outputs = torch.cat([outputs, o], dim=2)
        z = torch.matmul(outputs, self.wo)
        return z

    def init_weights(self):
        self.w.data.uniform_(-0.8, 0.8 )
        self.wo.data.uniform_(-1.0, 1.0)

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

class EMSA(nn.Module):

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, H=7, W=7, ratio=3, apply_transform=True):

        super(EMSA, self).__init__()
        self.H = H
        self.W = W
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.ratio = ratio
        if (self.ratio > 1):
            self.sr = nn.Sequential()
            self.sr_conv = nn.Conv2d(d_model, d_model, kernel_size=ratio + 1, stride=ratio, padding=ratio // 2,
                                     groups=d_model)
            self.sr_ln = nn.LayerNorm(d_model)

        self.apply_transform = apply_transform and h > 1
        if (self.apply_transform):
            self.transform = nn.Sequential()
            self.transform.add_module('conv', nn.Conv2d(h, h, kernel_size=1, stride=1))
            self.transform.add_module('softmax', nn.Softmax(-1))
            self.transform.add_module('in', nn.InstanceNorm2d(h))

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

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

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):

        b_s, nq, c = queries.shape
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)

        if (self.ratio > 1):
            x = queries.permute(0, 2, 1).view(b_s, c, self.H, self.W)  # bs,c,H,W
            x = self.sr_conv(x)  # bs,c,h,w
            x = x.contiguous().view(b_s, c, -1).permute(0, 2, 1)  # bs,n',c
            x = self.sr_ln(x)
            k = self.fc_k(x).view(b_s, -1, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, n')
            v = self.fc_v(x).view(b_s, -1, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, n', d_v)
        else:
            k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
            v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        if (self.apply_transform):
            att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, n')
            att = self.transform(att)  # (b_s, h, nq, n')
        else:
            att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, n')
            att = torch.softmax(att, -1)  # (b_s, h, nq, n')

        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
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
        # self.MultiHeadAttention = MultiHeadAttention(Attention_Dim, N_head, Input_dim=hidden_size * 2)
        # self.MultiHeadAttention = EMSA(d_model=200, d_k=200, d_v=200, h=7, H=10, W=5, ratio=2)
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


        # x = self.Embed(x).permute(0, 2, 1)




        out = [conv(x0) for conv in self.conv]

        # 这里如何写成自动化的？？
        fun0 = torch.nn.ZeroPad2d(padding=(0, self.Max_length - out[0].shape[2], 0, 0, 0, 0))
        fun1 = torch.nn.ZeroPad2d(padding=(0, self.Max_length - out[1].shape[2], 0, 0, 0, 0))
        fun2 = torch.nn.ZeroPad2d(padding=(0, self.Max_length - out[2].shape[2], 0, 0, 0, 0))


        x1 = torch.cat([fun0(out[0]), fun1(out[1]), fun2(out[2])], dim=1)

        x1 = self.Dropout(x1)
        merge = torch.cat([x0,x1],dim=1)

        x, _ = self.lstm(merge.permute(0, 2, 1))

        # x = self.MultiHeadAttention(x,x,x)
        x = self.MultiHeadAttention(x)

        x = self.Dense(x.flatten(1))

        return x


if __name__ == '__main__':
    model = Mymodel(
        vocab_size=21,
        Embed_D=51,
        pool_size=5,
        dropout=0.6,
        output_length=21,
        convDim=64,
        hidden_size=100,
        Attention_Dim = 21,
        Max_length = 50,
        N_head = 5
    )
    x = torch.randint(0, 21, [51, 50])
    y = model(x)
    print(y.shape)
