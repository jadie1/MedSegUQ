import torch

from models.subspaces import Subspace
# from ._assess_dimension import _infer_dimension_

import numpy as np

from scipy.special import gammaln 
import importlib
import torch
import torch.nn as nn
import numpy as np
from scipy.special import softmax

class Model(nn.Module):
    def __init__(self, args, subspace_kwargs=None,):
        super(Model, self).__init__()
        self.device = args.device
        model_module = importlib.import_module('.%s' % args.base_model_name, 'models')
        self.base_model = model_module.Model(args).to(self.device)
        self.base_model.load_state_dict(torch.load(args.base_model_path, map_location=self.device))

        self.num_parameters = sum(param.numel() for param in self.base_model.parameters())
        self.subspace_type = 'covariance'

        self.register_buffer('mean', torch.zeros(self.num_parameters))
        self.register_buffer('sq_mean', torch.zeros(self.num_parameters))
        self.register_buffer('n_models', torch.zeros(1, dtype=torch.long))

        # Initialize subspace
        if subspace_kwargs is None:
            subspace_kwargs = dict()
        self.subspace = Subspace.create(self.subspace_type, num_parameters=self.num_parameters,
                                        **subspace_kwargs)

        self.var_clamp = 1e-06
        self.cov_factor = None

    # def forward(self, *args, **kwargs):
    #     return self.base_model(*args, **kwargs)
    def forward(self, x: torch.Tensor, member_id=None) -> torch.Tensor:
        x = self.base_model(x, member_id)
        return x


    def collect_model(self, base_model, *args, **kwargs):
        # need to refit the space after collecting a new model
        self.cov_factor = None

        # w = flatten([param.detach().cpu() for param in base_model.parameters()])
        w = flatten([param for param in base_model.parameters()])
        # first moment
        self.mean.mul_(self.n_models.item() / (self.n_models.item() + 1.0))
        self.mean.add_(w / (self.n_models.item() + 1.0))
        print(self.mean.shape)

        # second moment
        self.sq_mean.mul_(self.n_models.item() / (self.n_models.item() + 1.0))
        self.sq_mean.add_(w ** 2 / (self.n_models.item() + 1.0))
        input(self.sq_mean.shape)

        dev_vector = w - self.mean

        self.subspace.collect_vector(dev_vector, *args, **kwargs)
        self.n_models.add_(1)

    def _get_mean_and_variance(self):
        variance = torch.clamp(self.sq_mean - self.mean ** 2, self.var_clamp)
        return self.mean, variance

    def fit(self):
        if self.cov_factor is not None:
            return
        self.cov_factor = self.subspace.get_space()

    def set_swa(self):
        print("Num models:", self.n_models.item())
        set_weights(self.base_model, self.mean, self.device)

    def sample(self, scale=0.5, diag_noise=True):
        self.fit()
        mean, variance = self._get_mean_and_variance()

        eps_low_rank = torch.randn(self.cov_factor.size()[0]).to(self.device)
        z = self.cov_factor.t() @ eps_low_rank
        if diag_noise:
            z += variance.sqrt() * torch.randn_like(variance)
        z *= scale ** 0.5
        sample = mean + z

        # apply to parameters
        set_weights(self.base_model, sample, self.device)
        return sample

    def get_space(self, export_cov_factor=True):
        mean, variance = self._get_mean_and_variance()
        if not export_cov_factor:
            return mean.clone(), variance.clone()
        else:
            self.fit()
            return mean.clone(), variance.clone(), self.cov_factor.clone()

    def infer_dimension(self, update_max_rank=True, use_delta=True):
        if use_delta:
            delta = self.subspace.delta

        _, var, subspace = self.get_space()
        subspace /= (self.n_models.item() - 1) ** 0.5
        tr_sigma = var.sum()

        spectrum, _ = torch.eig(subspace @ subspace.t())
        spectrum, _ = torch.sort(spectrum[:,0], descending=True)
        
        new_max_rank, ll, _ = _infer_dimension_(spectrum.numpy(), tr_sigma.numpy(),
                                                self.n_models.item(), self.num_parameters, delta)

        if new_max_rank + 1 == self.subspace.max_rank and update_max_rank:
            self.subspace.max_rank += 1


def _assess_dimension_(spectrum, unscaled_vhat, rank, n_samples, n_features, alpha = 1, beta = 1):
    """Compute the likelihood of a rank ``rank`` dataset
    The dataset is assumed to be embedded in gaussian noise of shape(n,
    dimf) having spectrum ``spectrum``.
    Parameters
    ----------
    spectrum : array of shape (n)
        Data spectrum.
    rank : int
        Tested rank value.
    n_samples : int
        Number of samples.
    n_features : int
        Number of features.
    Returns
    -------
    ll : float,
        The log-likelihood
    Notes
    -----
    This implements the method of `Thomas P. Minka:
    Automatic Choice of Dimensionality for PCA. NIPS 2000: 598-604`
    """
    if rank > len(spectrum):
        raise ValueError("The tested rank cannot exceed the rank of the"
                         " dataset")

    pu = -rank * np.log(2.)
    for i in range(rank):
        pu += (gammaln((n_features - i) / 2.) -
               np.log(np.pi) * (n_features - i) / 2.)
    #pu -= rank * gammaln(alpha/2) + gammaln(alpha * (n_features - rank)/2)
    #pu += alpha * (n_features - rank) / 2 * np.log(beta * (n_features - rank) / 2) + alpha * rank * np.log(beta / 2)

    pl = np.sum(np.log(spectrum[:rank]))
    pl = -pl * n_samples / 2.

    if rank == n_features:
        pv = 0
        v = 1
    else:
        #v = np.sum(spectrum[rank:]) / (n_features - rank)
        v = unscaled_vhat / (n_features - rank)
        #print(-np.log(v))
        pv = -np.log(v) * n_samples * (n_features - rank) / 2.

    m = n_features * rank - rank * (rank + 1.) / 2.
    pp = np.log(2. * np.pi) * (m + rank + 1.) / 2.

    pa = 0.
    spectrum_ = spectrum.copy()
    spectrum_[rank:n_features] = v
    for i in range(rank):
        for j in range(i + 1, len(spectrum)):
            #print((spectrum[i] - spectrum[j]) *
            #          (1. / spectrum_[j] - 1. / spectrum_[i]), i, j)
            pa += np.log((spectrum[i] - spectrum[j]) *
                      (1. / spectrum_[j] - 1. / spectrum_[i])) + np.log(n_samples)

    #print(pu, pl-pa/2., pv, pp)
    #pu = 0
    #pp = 0
    #pa = 0
    #pv = 0
    ll = pu + pl + pv + pp - pa / 2. - (rank + m) * np.log(n_samples) * 3 / 2.
    return ll

def _infer_dimension_(spectrum, tr_sigma, n_samples, n_features, delta = None, alpha = 1, beta = 1):
    """Infers the dimension of a dataset of shape (n_samples, n_features)
    The dataset is described by its spectrum `spectrum`.
    """
    n_spectrum = len(spectrum)
    ll = np.empty(n_spectrum)
    unscaled_vhat = np.empty(n_spectrum)
    for rank in range(n_spectrum):
        if delta is not None:
            unscaled_vhat[rank] = tr_sigma - (rank * delta / (n_samples - 1) + spectrum[:rank].sum())
            #print('unscaled_vhat is : ', unscaled_vhat)
        else:
            unscaled_vhat[rank] = tr_sigma - spectrum[:rank].sum()

        ll[rank] = _assess_dimension_(spectrum, unscaled_vhat[rank], rank, n_samples, n_features, alpha = alpha, beta = beta)
    return np.nanargmax(ll)+1, ll, unscaled_vhat

def flatten(lst):
    tmp = [i.contiguous().view(-1,1) for i in lst]
    return torch.cat(tmp).view(-1)

def set_weights(model, vector, device=None):
    offset = 0
    for param in model.parameters():
        param.data.copy_(vector[offset:offset + param.numel()].view(param.size()).to(device))
        offset += param.numel()