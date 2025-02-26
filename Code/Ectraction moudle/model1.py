import torch
import torch.nn as nn
import torch.nn.functional as F



class Transone(nn.Module):
    def __init__(self, n_in=7, n_hid=31, heads=7):
        super(Transone, self).__init__()

        self.linearq = nn.Linear(n_in,n_hid)
        self.lineark = nn.Linear(n_in, n_hid)
        self.linearv = nn.Linear(n_in, n_hid)

        self.linearq2 = nn.Linear(n_in, n_hid)
        self.lineark2 = nn.Linear(n_in, n_hid)

        self.linear1 = nn.Linear(n_hid, n_in)
        self.relu = nn.ReLU()
        self.heads = heads
        self.n_in = n_in
        self.head_dim = n_hid//heads
        self.layer = nn.LayerNorm(normalized_shape=n_hid)

    def forward(self, inputs):#(batch,c,t)

        #在时间维度上进行计算
        Q = self.linearq(inputs).reshape(inputs.shape[0], inputs.shape[1], self.heads,-1)
        K = self.lineark(inputs).reshape(inputs.shape[0], inputs.shape[1], self.heads,-1)
        V = self.linearv(inputs).reshape(inputs.shape[0], inputs.shape[1], self.heads,-1)

        Q = Q.permute(0,2,1,3)
        #(batch,head,c,t)
        K = K.permute(0,2,1,3)
        #(batch,head,c,t)
        V = V.permute(0, 2, 1, 3)
        #(batch,head,c,t)

        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim).float())
        attention_weights = F.softmax(attention_scores, dim=-1)
        # (batch,head,c,c)
        outputs = torch.matmul(attention_weights,V)# (batch,num,c,t)

        outputs = outputs.permute(0,2,1,3)# (batch,c,num,t)
        outputs = outputs.reshape(outputs.shape[0],outputs.shape[1],-1)#(batch,c,t)

        outputs = self.layer(outputs)
        outputs = self.linear1(outputs)
        outputs = self.relu(outputs)
        outputs = outputs + inputs#(batch,c,t)


        Q2 = self.linearq2(outputs)
        K2 = self.lineark2(outputs)

        attention_scores2 = torch.matmul(Q2, K2.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.n_in).float())
        attention_weights2 = F.softmax(attention_scores2, dim=-1)


        return outputs,attention_weights2#(batch,c,t)、(batch,c,c)



























