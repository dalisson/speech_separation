import torch
import torch.nn as nn
from torch.nn import LSTM

class BaseRNNBlock(nn.Module):
    '''
    O bloco impar passa a RNN ao longo da dimensão do tempo, ou seja,
    ao longo de R chunks de tamanho K
    '''
    
    def __init__(self, Parameter=128, hidden_size=128, **kwargs):
        super(BaseRNNBlock, self).__init__()
        self.lstm1 = LSTM(input_size=Parameter, hidden_size=hidden_size, batch_first=True, bidirectional=True, **kwargs)
        self.lstm2 = LSTM(input_size=Parameter, hidden_size=hidden_size, batch_first=True, bidirectional=True, **kwargs)
        #no paper P
        self.P = nn.Linear(hidden_size*2 + Parameter, Parameter)
        
    def forward(self, x):
        outp1, _ = self.lstm1(x)
        outp2, _ = self.lstm2(x)
        outp = torch.cat((outp1 * outp2, x), dim=-1)
        #O tensor volta a NxKxR
        return self.P(outp)

class RNNBlock(nn.Module):
    '''
    Contem um bloco par e um bloco impar
    '''
    
    def __init__(self, K, R, hidden_size=128,**kwargs):
        super(RNNBlock, self).__init__()
        self.oddBlock = BaseRNNBlock(Parameter=K, hidden_size=hidden_size, **kwargs)
        self.evenBlock = BaseRNNBlock(Parameter=R, hidden_size=hidden_size, **kwargs)
        
    def forward(self, x):
        outp = self.oddBlock(x)
        #skip connection
        x += outp
        #Tensor NxRxK -> NxKxR
        x = torch.transpose(x, 1,-1)
        outp = self.evenBlock(x)
        #skip connection
        x += outp
        #Tensor NxKxR -> NxRxK
        x = torch.transpose(x, 1,-1)
        return x

class Facebookmodel(nn.Module):
    
    def __init__(self, n, k, r, c=2, l=8, b=1, **kwargs):
        super(Facebookmodel, self).__init__()
        self.encoder = nn.Conv1d(1, n, l, int(l/2))
        self.rnnblocks = [RNNBlock(k, r, **kwargs) for _ in range(b)]
        self.d = nn.Conv1d(r, c*r, kernel_size=1)
        self.activation = torch.nn.PReLU(num_parameters=1, init=0.25)
        self.decoder = nn.ConvTranspose1d(64, 1, kernel_size=l, stride=int(l/2))
    
    def forward(self, x):
        encoded = self.encoder(x).squeeze(0)
        chunks = self.chunk(encoded)
        outps = list()
        for block in self.rnnblocks:
            res = block(chunks)
            res = self.d(self.activation(res))
            outps.append(res)
        
        s1, s2 = self.split_channels(outps)
        o1, o2 = self.apply_overlap_and_add(s1, s2)
        return self.decode(o1, o2)
    
    def split_channels(self, x):
        channel_1 = []
        channel_2 = []
        for o in x:
            divided = o.unfold(-2, 181, 181)
            t = torch.transpose(divided, -1, -2)
            channel_1.append(divided[:,0,...])
            channel_2.append(divided[:,1,...])
        return channel_1, channel_2
    
    def chunk(self, x):
        x = torch.cat((x, torch.zeros((64, 110))), dim=-1)
        x = torch.cat((torch.zeros((64, 89)), x), dim=-1)
        
        return x.unfold(-1, 178, 89)
    
    def decode(self, c1, c2):
        restored_1, restored_2 = list(), list()
        
        for a in c1:
            a = a[:, 89:-110].unsqueeze(0)
            a = self.decoder(a)
            restored_1.append(a)
        
        for a in c2:
            a = a[:,89:-110].unsqueeze(0)
            a = self.decoder(a)
            restored_2.append(a)
        
        return restored_1, restored_2
    
    def apply_overlap_and_add(self, channel_1, channel_2):
        overlapped_1 = list()
        overlapped_2 = list()
        
        for el in channel_1:
            r = self.overlap_and_add(el)
            overlapped_1.append(r)
        
        for el in channel_2:
            r = self.overlap_and_add(el)
            overlapped_2.append(r)
            
        return overlapped_1, overlapped_2
    
    def overlap_and_add(self, x):
        result = torch.nn.functional.fold(x, (1, 16198) ,kernel_size=(1,178), stride=(1,89))
        return result.squeeze(1).squeeze(1)