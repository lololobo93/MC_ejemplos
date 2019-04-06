########## ICTP-SAIFR Minicourse on Machine Learning for Many-Body Physics ##########
### Roger Melko, Juan Carrasquilla, Lauren Hayward Sierens and Giacomo Torlai
### Tutorial 4: Restricted Boltzmann Machine (RBM)
#####################################################################################

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import itertools as it
import numpy as np

class RBM(nn.Module):
    '''
    Restricted Boltzmann Machine

    Args:
        num_visible (int): number of visible nodes.
        num_hidden (int): number of hidden nodes.

    Attributes:
        W (2darray): weights.
        v_bias (1darray): bias for visible layer.
        h_bias (1darray): bias for hidden layer.
    '''

    def __init__(self, num_visible, num_hidden, num_samples = 500):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(num_hidden, num_visible) * 1e-2)
        self.v_bias = nn.Parameter(torch.zeros(num_visible))
        self.h_bias = nn.Parameter(torch.randn(num_hidden) * 1e-2)
        self.num_visible = num_visible
        self.num_hidden = num_hidden

        self.h_sample = (torch.FloatTensor(num_samples, num_hidden).uniform_() > 0.5).float()

    def _v_to_h(self, v):
        '''
        forward pass p(h|v) from visible to hidden, v is visible input.
        '''
        p_h = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        return p_h

    def _h_to_v(self, h):
        '''
        backward pass p(v|h) from hidden to visible, h is hidden input.
        '''
        p_v = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return p_v

    def contrastive_divergence(self, v, k):
        '''
        Args:
            v (ndarray): visible input.
            k (in): CD-k, means k times v->h & h->v sweep in a single contrastive divergence run.

        Returns:
            ndarray: visible obtained through CD sampling.
        '''
        prob_h = self._v_to_h(v)
        h = sample_from_prob(prob_h)
        for _ in range(k):
            prob_v = self._h_to_v(h)
            v = sample_from_prob(prob_v)
            prob_h = self._v_to_h(v)
            h = sample_from_prob(prob_h)
        return v

    def free_energy(self, v):
        '''
        free energy E(x) = -log(\sum_h exp(x, h)) = -log(p(x)*Z).
        It can be used to obtain negative log-likelihood L = <E(x)>_{data} - <E(x)>_{model}.

        Args:
            v (1darray,2darray): visible input with size ([batch_size, ]data_size).

        Return:
            float: the free energy loss.
        '''
        vbias_term = v.mv(self.v_bias)
        wx_b = F.linear(v, self.W, self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(dim=-1)
        return (-hidden_term - vbias_term).mean()
    
    def stochastic_maximum_likelihood(self, k):
        # type: (int) -> (tf.Tensor, tf.Tensor, tf.Tensor)
        h = self.h_sample
        for i in range(k):
            prob_v = self._h_to_v(h)
            v = sample_from_prob(prob_v)
            prob_h = self._v_to_h(v)
            h = sample_from_prob(prob_h)

        self.h_sample = h
        return self.h_sample, v

def sample_from_prob(prob_list):
    '''
    from probability to 0-1 sample.

    Args:
        prob_list (1darray): probability of being 1.

    Returns:
        1darray: 0-1 array.
    '''
    rand = torch.rand(prob_list.size())
    if prob_list.is_cuda:
        rand = rand.cuda()
    return (1+torch.sign(prob_list - rand))/2.
