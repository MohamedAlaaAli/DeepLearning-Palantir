import torch
import torch.nn as nn
import numpy as np

# reference 
# https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html


class CustomBatchNorm(nn.Module):
    def __init__(self, num_features, epsilon=1e-5, momentum=0.9):
        super(CustomBatchNorm, self).__init__()
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # Running estimates of mean and variance
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            
            x_normalized = (x - batch_mean) / torch.sqrt(batch_var + self.epsilon)
            out = self.gamma * x_normalized + self.beta

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
        else:
            x_normalized = (x - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)
            out = self.gamma * x_normalized + self.beta

        return out



class BatchNorm:
    def __init__(self, num_features, epsilon=1e-5, momentum=0.9):
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum

        # Learnable parameters
        self.gamma = np.ones((num_features,))
        self.beta = np.zeros((num_features,))

        # Running estimates of mean and variance
        self.running_mean = np.zeros((num_features,))
        self.running_var = np.ones((num_features,))

    def forward(self, x, training=True):
        if training:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            
            x_normalized = (x - batch_mean) / np.sqrt(batch_var + self.epsilon)
            out = self.gamma * x_normalized + self.beta

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
        else:
            x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            out = self.gamma * x_normalized + self.beta

        return out

    def backward(self, dout, x):
        N, D = dout.shape
        
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        
        x_mu = x - batch_mean
        std_inv = 1. / np.sqrt(batch_var + self.epsilon)
        
        x_normalized = (x - batch_mean) / np.sqrt(batch_var + self.epsilon)

        dx_normalized = dout * self.gamma
        dvar = np.sum(dx_normalized * x_mu, axis=0) * -.5 * std_inv**3
        dmean = np.sum(dx_normalized * -std_inv, axis=0) + dvar * np.mean(-2. * x_mu, axis=0)

        dx = (dx_normalized * std_inv) + (dvar * 2 * x_mu / N) + (dmean / N)
        dgamma = np.sum(dout * x_normalized, axis=0)
        dbeta = np.sum(dout, axis=0)

        return dx, dgamma, dbeta

np.random.seed(42)
x = np.random.randn(10, 5)  
bn = BatchNorm(num_features=5)

out = bn.forward(x, training=True)
print("Forward output (training):")
print(out)

out_inference = bn.forward(x, training=False)
print("Forward output (inference):")
print(out_inference)

# Backward pass
dout = np.random.randn(10, 5)  
dx, dgamma, dbeta = bn.backward(dout, x)
print("Backward output (dx, dgamma, dbeta):")
print(dx)
print(dgamma)
print(dbeta)
