import numpy as np
import torch
from torch import nn
from .sinkhornknopp import SemiCurrSinkhornKnopp_stable


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length

def set_rho(rho_strategy, current, total, rho_upper, rho_base):
    if rho_strategy == "sigmoid":
        rho = sigmoid_rampup(current, total)* rho_upper + rho_base
    elif rho_strategy == "linear":
        rho = current / total * rho_upper + rho_base
    else:
        raise NotImplementedError
    if rho > 1.0:
        rho = min(rho, 1.0)
    return rho
            
class OT_Attn(nn.Module):
    def __init__(self,impl="hot") -> None:
        super().__init__()
        self.impl = impl
        print("ot impl: ", impl)
    
    def normalize_feature(self,x):
        x = x - x.min(-1)[0].unsqueeze(-1)
        return x

    def OT(self, weight1, weight2, iterations, iterations_per_epoch):
        """
        Parmas:
            weight1 : (N, D)
            weight2 : (M, D)
        
        Return:
            flow : (N, M)
            dist : (1, )
        """

        if self.impl == "hot":
            rho_value = set_rho(rho_strategy="sigmoid", current=iterations, total=10*iterations_per_epoch, rho_upper=1, rho_base=0.1)
            sk = SemiCurrSinkhornKnopp_stable(num_iters=3, epsilon=0.1, gamma=1, stoperr=1e-10, numItermax=1000, rho=rho_value, semi_use=True, prior=None)

            cost = torch.cdist(weight1, weight2)
            cost = cost / cost.max()

            flow, _ = sk.cost_forward(cost, final=False, count=True)

            flow = flow / rho_value

            cost = cost.type(torch.FloatTensor).to(weight1.device)
            flow = flow.type(torch.FloatTensor).to(weight1.device)
            dist = cost * flow
            dist = torch.sum(dist)
            return flow, dist

        else:
            raise NotImplementedError

    def forward(self, x, y, iterations=0, iterations_per_epoch=20):
        '''
        x: (N, D)
        y: (M, D)
        '''
        x = self.normalize_feature(x)
        y = self.normalize_feature(y)
        
        pi, dist = self.OT(x, y, iterations, iterations_per_epoch)
        return pi, dist