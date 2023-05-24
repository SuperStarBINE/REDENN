import torch
import math
# 按差分方程计算指定nr的矩阵乘幂结果的(X^n)_{12}的方差
def get_variance(n,r):
    d_parms = [[1],
               [0,1],
               [8,1,1],
               [16,25,3,1],
               [260,153,65,6,1],
               [1856,21,629,137,10,1],
               [26240,26603,11253,1990,263,15,1]]
    parms= d_parms[n-1]
    res = 0
    for i,x in enumerate(parms):
        for j in range(i):
            x*=(r-j)
        res+=x
    return res

class REDE_BASIC(torch.nn.Module):
    def __init__(self, n) -> None:
        super().__init__()
        self.n = n
        
    def forward(self, M):
        # M (...,r,r)        
        Mn     = M
        cumsum = Mn
        Mn     = Mn
        for i in range(1, self.n):
            Mn     = (torch.matmul(Mn, M) / math.sqrt(M.shape[-1])) 
            cumsum = Mn + cumsum
        cumsum /= math.sqrt(self.n)
        return cumsum
    
class REDE_INFER(torch.nn.Module):
    def __init__(self, 
                 r,
                 n) -> None:
        super().__init__()
        assert n<=7
        self.r, self.n = r, n
        self.mask         = torch.arange(0,self.r,1)
        self.decays       = [1]+[math.sqrt(get_variance(i+1,r)/get_variance(i,r)) for i in range(1,self.n)]
        self.decay_cumsum = math.sqrt(self.n)
        
    def forward(self, M):
        # M (...,r,r)
        shape = M.shape[:-1]
        M = M.view(shape + (self.r, self.r))
        
        Mn     = M
        cumsum = Mn.clone()
        Mn.div_(self.decays[0])
        for i in range(1, self.n):
            Mn = torch.matmul(Mn, M).div_(self.decays[i]) # ...,r,r
            cumsum.add_(Mn)
        cumsum[...,self.mask,self.mask].zero_()
        cumsum.div_(self.decay_cumsum)
        cumsum = cumsum.view(shape+(self.r*self.r,))
        return cumsum
    
class REDE(torch.nn.Module):
    def __init__(self, 
                 r,
                 n) -> None:
        super().__init__()
        assert n<=7
        self.r, self.n = r, n
        self.mask         = torch.arange(0,self.r,1)
        self.decays       = [1]+[math.sqrt(get_variance(i+1,r)/get_variance(i,r)) for i in range(1,self.n)]
        self.decay_cumsum = math.sqrt(self.n)

    def forward(self, M):
        # M (...,r*r)
        shape = M.shape[:-1]
        M = M.view(shape + (self.r, self.r))
        
        Mn     = M
        cumsum = Mn
        Mn     = Mn / self.decays[0]
        for i in range(1, self.n):
            Mn     = (torch.matmul(Mn, M) / self.decays[i]) # ...,r,r
            cumsum = Mn + cumsum
        cumsum[...,self.mask,self.mask]=0
        cumsum/= self.decay_cumsum
        cumsum = cumsum.view(shape+(self.r*self.r,))
        return cumsum

class REDE_GROUP(torch.nn.Module):
    def __init__(self, 
                 r,
                 n,
                 group=1) -> None:
        super().__init__()
        assert n<=7
        self.r = r
        self.group = group
        self.rede = REDE(r,n)
        
    def forward(self, M):
        # M (...,g*r*r)
        shape = M.shape[:-1]
        M = M.view(shape + (self.group, self.r*self.r))
        M = self.rede(M)
        M = M.view(shape + (self.group*self.r*self.r, ))
        return M

if __name__=="__main__":
    b,r = 100,100

    if True:
        X = torch.randn((b,1,2*r*r)).to('cuda')
        model = REDE(r=r,n=7).cuda()
        model = REDE_INFER(r=r,n=7).cuda()
        model = REDE_GROUP(r=r,n=7,group=2).cuda()
        y = model(X)
        print(y.shape)
        print(y.mean())
        print(y.min())
        print(y.max())
        print(y.var())
        print(y.shape)