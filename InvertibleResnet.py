import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np 
from SpectralNormGouk import *

logabs = lambda x: torch.log(torch.abs(x))
class ActNorm(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
    
        self.loc = nn.Parameter(torch.zeros(1, in_channel))
        self.scale = nn.Parameter(torch.ones(1, in_channel))
        self.initialized = False

    def initialize(self, input):
        with torch.no_grad():
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

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, logdet=False, reverse=False):
        if reverse:
            return input / self.scale - self.loc

        if not self.initialized:
            self.initialize(input)
            self.initialized = True

        log_abs = logabs(self.scale)
        dlogdet = torch.sum(log_abs)

        if logdet:
            return self.scale * (input + self.loc), dlogdet

        else:
            return self.scale * (input + self.loc)


def vjp(ys, xs, v):
    vJ = torch.autograd.grad(ys, xs, grad_outputs=v, create_graph=True, retain_graph=True, allow_unused=True)
    return tuple([j for j in vJ])

class block(nn.Module):
    def __init__(self, dim, hidden_dim, reverse_iterations=40, magnitude=0.7, bias=False, activation=True, simple=False, n_power_iterations=5):
        super(block, self).__init__()
        self.dim = dim
        self.reverse_iterations = reverse_iterations
        self.activation=activation
        self.net1 = spectral_norm(nn.Linear(dim, hidden_dim, bias=bias), n_power_iterations=n_power_iterations, magnitude=magnitude)
        self.net2 = spectral_norm(nn.Linear(hidden_dim,hidden_dim, bias=bias), n_power_iterations=n_power_iterations, magnitude=magnitude)
        self.net3 = spectral_norm(nn.Linear(hidden_dim,hidden_dim, bias=bias), n_power_iterations=n_power_iterations, magnitude=magnitude)
        self.net4 = spectral_norm(nn.Linear(hidden_dim,hidden_dim, bias=bias), n_power_iterations=n_power_iterations, magnitude=magnitude)
        self.net5 = spectral_norm(nn.Linear(hidden_dim, dim, bias=bias), n_power_iterations=n_power_iterations, magnitude=magnitude)  
        self.elu = nn.ELU()
        self.normalize()
        self.simple = simple
    def normalize(self):
        for net in [self.net1, self.net2, self.net3, self.net4, self.net5]:
            for k,hook in net._forward_pre_hooks.items():
                if isinstance(hook,SpectralNorm):
                    hook(net, None)
    def calcG(self, x):
        if self.simple: return self.net1(x)
        for net in [self.net1, self.net2, self.net3, self.net4, self.net5]:
            x = net(self.elu(x))        
        return x
    def forward(self, x, logdet=False, reverse=False, num_logdet_iter=1,power_seq_len=10):
        if reverse:
            y = x
            for count in range(self.reverse_iterations):
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
            
            
            
class InvertibleResnet(nn.Module):
    def __init__(self, dim, num_blocks,magnitude, hidden_dim=600, reverse_iterations=40):
        super(InvertibleResnet, self).__init__()
        l = []
        self.num_blocks = num_blocks
        self.dim = dim
        self.reverse_iterations = reverse_iterations
        for i in range(0,num_blocks):
            b = block(dim, magnitude=magnitude, reverse_iterations=reverse_iterations, hidden_dim=hidden_dim)
            b.normalize()
            l.append(b)
            l.append(ActNorm(dim))
        self.net = nn.Sequential(*l)
    def forward(self, x, logdet=False, reverse=False, num_logdet_iter=1):
        if reverse:
            for module in self.net[::-1]:
                x = module(x, reverse=True)
            return x
        else:
            if logdet:
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
