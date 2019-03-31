import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np 
from SpectralNormGouk import *
def run_module_with_logdet(module, x, num_logdet_iter):
    if 'block' in str(module.__class__):
        x, dlogdet = module(x, logdet=True, reverse=False, num_logdet_iter=num_logdet_iter)
    else:
        x, dlogdet = module(x, logdet=True, reverse=False)
    return x, dlogdet
logabs = lambda x: torch.log(torch.abs(x))
class ActNorm(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
    
        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))
        self.initialized = False

    def initialize(self, input):
        with torch.no_grad():
            if len(input.shape) == 2: # linear
                flatten = input.permute(1, 0).contiguous().view(input.shape[1], -1)
                mean = (
                    flatten.mean(1)
                    .unsqueeze(1)
                    .permute(1, 0,)
                )
                std = (
                    flatten.std(1)
                    .unsqueeze(1)
                    .permute(1, 0)
                )
                self.loc.data.copy_(-mean.view_as(self.loc))
                self.scale.data.copy_(1 / (std.view_as(self.scale) + 1e-6))
            elif len(input.shape) == 4: # conv
                flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
                mean = (
                    flatten.mean(1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .permute(1, 0, 2, 3)
                )
                std = (
                    flatten.std(1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .permute(1, 0, 2, 3)
                )

                self.loc.data.copy_(-mean)
                self.scale.data.copy_(1 / (std + 1e-6))
            else:
                raise 'Input shape not supported {}'.format(input.shape)
    
    def forward(self, input, logdet=False, reverse=False):
        
        scale = self.scale if len(input.shape) == 4 else self.scale.view(1, -1)
        loc = self.loc if len(input.shape) == 4 else self.loc.view(1, -1)
        
        if reverse:
            return input / scale - loc

        if not self.initialized:
            self.initialize(input)
            self.initialized = True
        
        
        log_abs = logabs(scale) #return logdet PER sample, not adjusting for number of dimensions! 
        
        if len(input.shape) == 4:
            _, _, height, width = input.shape
        else:
            height, width = 1,1
        dlogdet = height * width * torch.sum(log_abs)           
        
        
        if logdet:
            return scale * (input + loc), dlogdet
        else:
            return scale * (input + loc)

def vjp(ys, xs, v):
    vJ = torch.autograd.grad(ys, xs, grad_outputs=v, create_graph=True, retain_graph=True, allow_unused=True)
    return tuple([j for j in vJ])

class block(nn.Module):
    def __init__(self, net, reverse_iterations=40):
        super(block, self).__init__()
        self.reverse_iterations = reverse_iterations
        self.net = net # residual neural network 
        self.normalize(self.net)
    def normalize(self, net):
        for n in net.modules():
            for k,hook in n._forward_pre_hooks.items():
                if isinstance(hook,SpectralNorm):
                    hook(n, None)
    def calcG(self, x):
        return self.net(x)
    def forward(self, x, logdet=False, reverse=False, num_logdet_iter=1,power_seq_len=10, reverse_iterations=None):
        if reverse:
            y = x
            for count in range(reverse_iterations if reverse_iterations else self.reverse_iterations):
                x = y - self.calcG(x)
            return x
        else:            
            if logdet:
                g = self.calcG(x) 
                y = g + x
                temp_training = self.training
                self.eval()
                logdet = 0    
                for i in range(0,num_logdet_iter): 
                    v = y.detach().clone().normal_()
                    w = v                            
                    for k in range(1,power_seq_len):
                        w = vjp(g,x,w)[0]
                        logdet += (-1)**(k+1) * torch.dot(w.flatten(), v.flatten()) / k
                logdet /= num_logdet_iter
                if temp_training:
                    self.train()
                return y, logdet/ y.shape[0]
            else:
                y = self.calcG(x) + x
                return y
def apply_module_reverse(module, x, reverse_iterations=None):
    if 'block' in str(module.__class__):
        return module(x, reverse=True, reverse_iterations=reverse_iterations)
    else:
        return module(x, reverse=True)
            
            
class InvertibleResnetLinear(nn.Module):
    def __init__(self, dim, num_blocks,magnitude=0.7, hidden_dim=600, reverse_iterations=40, bias=False, n_power_iterations=5):
        super(InvertibleResnetLinear, self).__init__()
        l = []
        self.num_blocks = num_blocks
        self.dim = dim
        self.reverse_iterations = reverse_iterations
        for i in range(0,num_blocks):
            
            net = nn.Sequential(nn.ELU(),
                                spectral_norm(nn.Linear(dim, hidden_dim, bias=bias), n_power_iterations=n_power_iterations, magnitude=magnitude),
                                nn.ELU(),
                                spectral_norm(nn.Linear(hidden_dim,hidden_dim, bias=bias), n_power_iterations=n_power_iterations, magnitude=magnitude),
                                nn.ELU(),
                                spectral_norm(nn.Linear(hidden_dim,hidden_dim, bias=bias), n_power_iterations=n_power_iterations, magnitude=magnitude),
                                nn.ELU(),
                                spectral_norm(nn.Linear(hidden_dim,hidden_dim, bias=bias), n_power_iterations=n_power_iterations, magnitude=magnitude),
                                nn.ELU(),
                                spectral_norm(nn.Linear(hidden_dim, dim, bias=bias), n_power_iterations=n_power_iterations, magnitude=magnitude))        
            
            b = block(net, reverse_iterations=reverse_iterations)
            l.append(b)
            l.append(ActNorm(dim))
        self.net = nn.Sequential(*l)
    def forward(self, x, return_logdet=False, reverse=False, num_logdet_iter=1, reverse_iterations=None):
        if reverse:
            for module in self.net[::-1]:
                x = apply_module_reverse(module, x, reverse_iterations)
            return x
        else:
            if return_logdet:
                logdet = 0                
                x.requires_grad =True
                for module in self.net:
                    if 'block' in str(module.__class__):
                        x, dlogdet = module(x, logdet=True, reverse=False, num_logdet_iter=num_logdet_iter)
                    else:
                        x, dlogdet = module(x, logdet=True, reverse=False)
                    logdet += dlogdet
                return x, logdet
            else:
                return self.net(x)
class SqueezeLayer(nn.Module):
    def __init__(self):
        super(SqueezeLayer, self).__init__()
    def forward(self, x, reverse=False, logdet=None):
        assert len(x.shape) == 4, 'Input must be 4dim, currently {}'.format(x.shape)
        if reverse:
            ret = x.view(x.shape[0], x.shape[1]//2//2, 2, 2, x.shape[2], x.shape[3]).permute(0,1,4,2,5,3).contiguous().view(x.shape[0], x.shape[1]//2//2, x.shape[2] * 2, x.shape[3] * 2)
        else:
            ret = x.view(x.shape[0], x.shape[1], x.shape[2] // 2, 2, x.shape[3] // 2, 2).permute(0,1,3,5,2,4).contiguous().view(x.shape[0], x.shape[1]*2*2, x.shape[2] // 2, x.shape[3] // 2)
        if logdet:
            return ret, 0
        else:
            return ret



        
class InvertibleResnetConv(nn.Module):
    def __init__(self, dim, hidden_dim = 32, list_num_blocks=(32,32,32),magnitude=0.7, reverse_iterations=10, bias=False, n_power_iterations=5):
        super(InvertibleResnetConv, self).__init__()
        
        self.dim = dim
        self.reverse_iterations = reverse_iterations        
        self.nets = nn.ModuleList()        
        for num_blocks in list_num_blocks:    
            l = nn.ModuleList()
            for i in range(0,num_blocks):
                net = nn.Sequential(nn.ELU(),
                                    spectral_norm(nn.Conv2d(dim, hidden_dim, 3, padding=1, bias=bias), n_power_iterations=n_power_iterations, magnitude=magnitude),
                                    nn.ELU(),
                                    spectral_norm(nn.Conv2d(hidden_dim, hidden_dim, 1, bias=bias), n_power_iterations=n_power_iterations, magnitude=magnitude),
                                    nn.ELU(),
                                    spectral_norm(nn.Conv2d(hidden_dim, dim, 3, padding=1, bias=bias), n_power_iterations=n_power_iterations, magnitude=magnitude),
                                   )        
                b = block(net, reverse_iterations=reverse_iterations)            
                l.append(ActNorm(dim))
                l.append(b)
            l.append(SqueezeLayer())    
            dim *= 2            
            self.nets.append(nn.Sequential(*l))
        
    def forward(self, x_list, return_logdet=False, reverse=False, num_logdet_iter=1,reverse_iterations=None):
        if reverse:
            for i, net in enumerate(self.nets[::-1]):
                if i == 0:
                    x = x_list[len(self.nets)-1-i]
                else:
                    x = torch.cat([x, x_list[len(self.nets)-1-i]], dim=1)                
                for module in net[::-1]:
                    x = apply_module_reverse(module, x, reverse_iterations)
            return x
        else:
            logdet = 0     
            y_list = []
            x = x_list
            
            if return_logdet:                
                x.requires_grad =True
            for i, net in enumerate(self.nets):                
                if return_logdet:  
                    for module in net:
                        x, dlogdet = run_module_with_logdet(module, x, num_logdet_iter)
                        logdet += dlogdet
                else:                    
                    x = net(x)  
                if i < len(self.nets)-1: 
                    y_list.append(x[:,x.shape[1]//2:])
                    x = x[:,:x.shape[1]//2] 
            y_list.append(x)
            if return_logdet:
                return y_list, logdet
            else:
                return y_list