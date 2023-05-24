import torch
import math
from models.REDE import REDE

class REDENN(torch.nn.Module):
    def __init__(self, 
                 in_features, 
                 out_features, 
                 r, 
                 n,
                 use_tanh=False) -> None:
        super().__init__()
        self.r                  = r
        self.n                  = n
        self.in_features        = in_features
        self.out_features       = out_features
        self.use_tanh = use_tanh
        
        if use_tanh: self.tanh = torch.nn.Tanh()
        
        self.W_linear   = torch.nn.Linear(in_features, r**2, bias=False)
        self.bn_1       = torch.nn.BatchNorm1d(r**2)
        self.activation = REDE(r,n)
        self.bn_2       = torch.nn.BatchNorm1d(r**2)
        self.M_weight   = torch.nn.Linear(r**2, out_features, bias=True)
        
    def forward(self, x):
        # x: (..., in_features)
        # y: (..., out_features)
        # print((x.mean(),x.var(),x.min(),x.max()))
        # print(x.shape)
        x = self.W_linear(x)
        x = self.bn_1(x)
        if self.use_tanh: x = self.tanh(x)
        x = self.activation(x) # ..., out_features
        x = self.bn_2(x)
        x = self.M_weight(x)
        
        return x

if __name__=="__main__":
    b,c = 100,10000

    if True:
        X = torch.randn((b,c)).to('cuda')
        model = REDENN(in_features=c, out_features=1000,r=40,n=6).cuda()
        y = model(X)
        print(y.mean())
        print(y.min())
        print(y.max())
        print(y.var())
        print(y.shape)