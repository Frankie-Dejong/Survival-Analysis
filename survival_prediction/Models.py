import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnPooling(nn.Module):
    def __init__(
        self, 
        L=128,
        D=2048,
        K=1
    ):
        super(AttnPooling, self).__init__()

        self.L = L
        self.D = D
        self.K = K
        
        self.V = nn.Linear(self.D, self.L)
        self.W = nn.Linear(self.L, self.K)


    def forward(self, x):
        '''
        alpha_k = exp(w^T tanh(V * h_k^T)) / sum_{j=1}^K exp(w^T tanh(V * h_j^T))
                = softmax(w^T tanh(V * h_k^T))
        return sum_{j=1}^K alpha_k * h_j as the attn score of this bag
        '''
        ori_x = x
        n, dim = x.shape
        
        assert dim == self.D
        
        x = self.V(x) # n * L
        x = torch.tanh(x) 
        x = self.W(x) # n * K
        
        alpha_k = F.softmax(x, dim=0) # n * K
        
        assert alpha_k.shape == (n, self.K)
        return torch.sum(alpha_k * ori_x, dim=0) # D



class Embedder(nn.Module):
    def __init__(self, input_dim=2048, output_dim=5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dense1 = nn.Linear(input_dim, output_dim)
        self.activate = nn.Softplus()

    def forward(self, x):
        x = self.dense1(x)
        x = self.activate(x)
        return x
    

class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class CancerPredictModel(nn.Module):
    def __init__(self, L=512, D=2048, K=1, output_dim=1):
        super().__init__()
        self.pooling = AttnPooling(
            L, D, K
        )
        self.get_alpha = Embedder(
            D, output_dim
        )
        self.get_beta = Embedder(
            D, output_dim
        )
        
    
    def forward(self, x):
        x = self.pooling(x)
        alpha = self.get_alpha(x)
        beta = self.get_beta(x)
        return alpha, beta

