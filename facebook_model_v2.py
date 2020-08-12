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

class FacebookModel(nn.Module):
    '''
    Modelo do facebook v2
    '''
    def __init__(self, n, k, r, c=2, l=8, b=1, **kwargs):
        super(FacebookModel, self).__init__()
        self.c = c
        self.encoder = nn.Conv1d(1, n, l, int(l/2))
        self.rnnblocks = [RNNBlock(k, r, **kwargs) for _ in range(b)]
        #register the layers into the class
        for i, block in enumerate(self.rnnblocks):
            nn.Module.add_module(self, 'rnn_block_%s' % i, block)
        self.d = nn.Conv1d(r, c*r, kernel_size=1)
        self.activation = torch.nn.PReLU(num_parameters=1, init=0.25)
        self.decoder = nn.ConvTranspose1d(n, 1, kernel_size=l, stride=int(l/2))

        #teste de decode com uma convolucao 2d
        self.decoder2d = nn.ConvTranspose2d(n, 1, kernel_size=(1, l), stride=(1, int(l/2)))
    
    def forward(self, x):
        encoded = self.encoder(x).squeeze(0)
        chunks = self.chunk(encoded)
        outps = list()
        for block in self.rnnblocks:
            res = block(chunks)
            res = self.d(self.activation(res))
            outps.append(res)
        
        outps = self.apply_overlap_and_add(outps)
        return self.decode2d(outps)

    
    def chunk(self, x):
        x = torch.cat((x, torch.zeros((64, 110))), dim=-1)
        x = torch.cat((torch.zeros((64, 89)), x), dim=-1)
        
        return x.unfold(-1, 178, 89)
        
    def decode2d(self, x):
        '''
        Testar o decode com uma convolucao de 2 dimensoes, audios das c fontes
        juntos no mesmo tensor
        '''
        
        restored = []
        for a in x:
            a = a[...,89:-110].unsqueeze(0)
            d = self.decoder2d(a).squeeze(1)
            restored.append(d)
        
        return restored
    
    def decode(self, x):
        '''
        Decode de separacao com convolucao 1d, os audios das c fontes diferentes sao separados
        previamente
        '''
        restored = [[] for _ in range(self.c)]
        
        for a in x:
            for i in range(self.c):
                t = a[:,i,89:-110].unsqueeze(0)
                t = self.decoder(t)
                restored[i].append(t.squeeze(0))
        
        return restored
    
    def apply_overlap_and_add(self, x):
        overlapped_added = list()
        for el in x:
            result = self.overlap_and_add(el)
            overlapped_added.append(result)
            
        return overlapped_added
    
    def overlap_and_add(self, x):
        '''
        Faz overlap and add usando pytorch fold
        '''
        x = torch.transpose(x, -2, -1)
        result = torch.nn.functional.fold(x, (self.c, 16198) ,kernel_size=(1,178), stride=(1,89))
        return result.squeeze(1)