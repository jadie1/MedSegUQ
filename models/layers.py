import torch
from torch import nn
import torch.nn.functional as F
import sys
import math
import json
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable
from numbers import Number
from torch.distributions.multivariate_normal import MultivariateNormal

"""
Batch ensemble conv 3d
"""
class BatchEnsemble_Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, num_members=4, transpose=False, **kwargs):
        super(BatchEnsemble_Conv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_members = num_members 
        if transpose:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels, **kwargs)
        else:
            self.conv = nn.Conv3d(in_channels, out_channels, **kwargs)

        # TODO: pass these as args rather than had set
        self.train_gamma = True
        self.random_sign_init = True
        self.constant_init = False
        self.probability = 0.5
        self.mixup = False
        
        self.alpha = nn.Parameter(torch.Tensor(num_members, in_channels))
        if self.train_gamma:
            self.gamma = nn.Parameter(torch.Tensor(num_members, out_channels))
        if kwargs['bias']:
            self.bias = nn.Parameter(torch.Tensor(self.num_members, out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.first_layer = True if in_channels == 1 else False

    def reset_parameters(self):
        if self.constant_init:
            nn.init.constant_(self.alpha, 1.)
            if self.random_sign_init:
                if self.probability  == -1:
                    with torch.no_grad():
                        factor = torch.ones(
                            self.num_members, device=self.alpha.device).bernoulli_(0.5)
                        factor.mul_(2).add_(-1)
                        self.alpha.data = (self.alpha.t() * factor).t()
                        if self.train_gamma:
                            self.gamma_mean.fill_(1.)
                            self.gamma_mean.data = (self.gamma_mean.t() * factor).t()
                elif self.probability  == -2:
                    with torch.no_grad():
                        positives_num = self.num_members // 2
                        factor1 = torch.Tensor([1 for i in range(positives_num)])
                        factor2 = torch.Tensor(
                            [-1 for i in range(self.num_members-positives_num)])
                        factor = torch.cat([factor1, factor2]).to(self.alpha.device)
                        self.alpha.data = (self.alpha.t() * factor).t()
                        if self.train_gamma:
                            self.gamma_mean.fill_(1.)
                            self.gamma_mean.data = (self.gamma_mean.t() * factor).t()
                else:
                    with torch.no_grad():
                        self.alpha.bernoulli_(self.probability)
                        self.alpha.mul_(2).add_(-1)
                        if self.train_gamma:
                            self.gamma_mean.bernoulli_(self.probability)
                            self.gamma_mean.mul_(2).add_(-1)
        else:
            nn.init.normal_(self.alpha, mean=1., std=0.5)
            #nn.init.normal_(self.alpha, mean=1., std=1)
            if self.train_gamma:
                nn.init.normal_(self.gamma, mean=1., std=0.5)
                #nn.init.normal_(self.gamma, mean=1., std=1)
            if self.random_sign_init:
                with torch.no_grad():
                    alpha_coeff = torch.randint_like(self.alpha, low=0, high=2)
                    alpha_coeff.mul_(2).add_(-1)
                    self.alpha *= alpha_coeff
                    if self.train_gamma:
                        gamma_coeff = torch.randint_like(self.gamma, low=0, high=2)
                        gamma_coeff.mul_(2).add_(-1)
                        self.gamma *= gamma_coeff
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def update_indices(self, indices):
        self.indices = indices

    def forward(self, x):
        # if not self.training and self.first_layer:
        #     # Repeated pattern in test: [[A,B,C],[A,B,C]]
        #     x = torch.cat([x for i in range(self.num_members)], dim=0)
        if self.train_gamma:
            num_examples_per_model = int(x.size(0) / self.num_members)
            extra = x.size(0) - (num_examples_per_model * self.num_members)

            if self.mixup:
                # Repeated pattern: [[A,A],[B,B],[C,C]]
                if num_examples_per_model != 0:
                    lam = np.random.beta(0.2, 0.2)
                    i = np.random.randint(self.num_members)
                    j = np.random.randint(self.num_members)
                    alpha = (lam * self.alpha[i] + (1 - lam) * self.alpha[j]).unsqueeze(0)
                    gamma = (lam * self.gamma[i] + (1 - lam) * self.gamma[j]).unsqueeze(0)
                    bias = (lam * self.bias[i] + (1 - lam) * self.bias[j]).unsqueeze(0)
                    for index in range(x.size(0)-1):
                        lam = np.random.beta(0.2, 0.2)
                        i = np.random.randint(self.num_members)
                        j = np.random.randint(self.num_members)
                        next_alpha = (lam * self.alpha[i] + (1 - lam) * self.alpha[j]).unsqueeze(0)
                        alpha = torch.cat([alpha,next_alpha], dim=0)
                        next_gamma = (lam * self.gamma[i] + (1 - lam) * self.gamma[j]).unsqueeze(0)
                        gamma = torch.cat([gamma,next_gamma], dim=0)
                        next_bias = (lam * self.bias[i] + (1 - lam) * self.bias[j]).unsqueeze(0)
                        bias = torch.cat([bias, next_bias], dim=0)
                else:
                    print("Error: TODO")
            else:
                # Repeated pattern: [[A,A],[B,B],[C,C]]
                alpha = torch.cat(
                    [self.alpha for i in range(num_examples_per_model)],
                    dim=1).view([-1, self.in_channels])
                gamma = torch.cat(
                    [self.gamma for i in range(num_examples_per_model)],
                    dim=1).view([-1, self.out_channels])
                if self.bias is not None:
                    bias = torch.cat(
                        [self.bias for i in range(num_examples_per_model)],
                        dim=1).view([-1, self.out_channels])
                    

            alpha.unsqueeze_(-1).unsqueeze_(-1).unsqueeze_(-1)
            gamma.unsqueeze_(-1).unsqueeze_(-1).unsqueeze_(-1)
            bias.unsqueeze_(-1).unsqueeze_(-1).unsqueeze_(-1)

            if extra != 0:
                alpha = torch.cat([alpha, alpha[:extra]], dim=0)
                gamma = torch.cat([gamma, gamma[:extra]], dim=0)
                if self.bias is not None:
                    bias = torch.cat([bias, bias[:extra]], dim=0)

            result = self.conv(x*alpha)*gamma

            return result + bias if self.bias is not None else result
        else:
            num_examples_per_model = int(x.size(0) / self.num_members)
            # Repeated pattern: [[A,A],[B,B],[C,C]]
            alpha = torch.cat(
                [self.alpha for i in range(num_examples_per_model)],
                dim=1).view([-1, self.in_channels])
            alpha.unsqueeze_(-1).unsqueeze_(-1)

            if self.bias is not None:
                bias = torch.cat(
                    [self.bias for i in range(num_examples_per_model)],
                    dim=1).view([-1, self.out_channels])
                bias.unsqueeze_(-1).unsqueeze_(-1)
            result = self.conv(x*alpha)
            return result + bias if self.bias is not None else result


class BE_Conv3d(BatchEnsemble_Conv3d):
    def __init__(self, in_channels, out_channels, num_members=4, **kwargs):
        super(BE_Conv3d, self).__init__(
            in_channels,
            out_channels,
            num_members=num_members,
            transpose=False,
            **kwargs)


class BE_ConvTranspose3d(BatchEnsemble_Conv3d):
    def __init__(self, in_channels, out_channels, num_members=4, **kwargs):
        super(BE_ConvTranspose3d, self).__init__(
            in_channels,
            out_channels,
            num_members=num_members,
            transpose=True,
            **kwargs)

"""
Rank one BNN conv3d
"""
class Rank_One_BNN_Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, num_members=4, transpose=False, **kwargs):
        super(Rank_One_BNN_Conv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_members = num_members 
        if transpose:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels, **kwargs)
        else:
            self.conv = nn.Conv3d(in_channels, out_channels, **kwargs)

        # TODO: pass these as args rather than had set
        self.random_sign_init = True
        self.constant_init = False
        self.probability = 0.5
        self.hidden_size=32
        self.kld_loss = 0
        self.kld_weight = 1e-6 # TODO
        self.train_gamma= True

        
        self.alpha_mean = nn.Parameter(torch.Tensor(num_members, in_channels))
        self.alpha_logvar = nn.Parameter(torch.Tensor(num_members, in_channels))


        self.gamma_mean = nn.Parameter(torch.Tensor(num_members, out_channels))
        self.gamma_logvar = nn.Parameter(torch.Tensor(num_members, out_channels))
        if kwargs['bias']:
            self.bias = nn.Parameter(torch.Tensor(self.num_members, out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.first_layer = True if in_channels == 1 else False

    def reset_parameters(self):
        if self.constant_init:
            nn.init.constant_(self.alpha, 1.)
            if self.random_sign_init:
                if self.probability  == -1:
                    with torch.no_grad():
                        factor = torch.ones(
                            self.num_members, device=self.alpha_mean.device).bernoulli_(0.5)
                        factor.mul_(2).add_(-1)
                        self.alpha_mean.data = (self.alpha_mean.t() * factor).t()
                        if self.train_gamma:
                            self.gamma_mean.fill_(1.)
                            self.gamma_mean.data = (self.gamma_mean.t() * factor).t()
                elif self.probability  == -2:
                    with torch.no_grad():
                        positives_num = self.num_members // 2
                        factor1 = torch.Tensor([1 for i in range(positives_num)])
                        factor2 = torch.Tensor(
                            [-1 for i in range(self.num_members-positives_num)])
                        factor = torch.cat([factor1, factor2]).to(self.alpha_mean.device)
                        self.alpha_mean.data = (self.alpha_mean.t() * factor).t()
                        if self.train_gamma:
                            self.gamma_mean.fill_(1.)
                            self.gamma_mean.data = (self.gamma_mean.t() * factor).t()
                else:
                    with torch.no_grad():
                        self.alpha_mean.bernoulli_(self.probability)
                        self.alpha_mean.mul_(2).add_(-1)
                        if self.train_gamma:
                            self.gamma_mean.bernoulli_(self.probability)
                            self.gamma_mean.mul_(2).add_(-1)
        else:
            nn.init.normal_(self.alpha_mean, mean=1., std=0.5)
            nn.init.normal_(self.alpha_logvar, mean=0., std=0.05)
            #nn.init.normal_(self.alpha, mean=1., std=1)
            nn.init.normal_(self.gamma_mean, mean=1., std=0.5)
            nn.init.normal_(self.gamma_logvar, mean=0., std=0.05)

            #nn.init.normal_(self.gamma, mean=1., std=1)
            if self.random_sign_init:
                with torch.no_grad():
                    alpha_coeff = torch.randint_like(self.alpha_mean, low=0, high=2)
                    alpha_coeff.mul_(2).add_(-1)
                    self.alpha_mean *= alpha_coeff
                    gamma_coeff = torch.randint_like(self.gamma_mean, low=0, high=2)
                    gamma_coeff.mul_(2).add_(-1)
                    self.gamma_mean *= gamma_coeff
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def update_indices(self, indices):
        self.indices = indices
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def forward(self, x):  
       # Sample
        alpha_sample = self.reparameterize(self.alpha_mean, self.alpha_logvar)
        gamma_sample = self.reparameterize(self.gamma_mean, self.gamma_logvar)

        if self.training:
            alpha_KLD = -0.5 * torch.sum(1 + self.alpha_logvar - self.alpha_mean.pow(2) - self.alpha_logvar.exp())
            gamma_KLD = -0.5 * torch.sum(1 + self.gamma_logvar - self.gamma_mean.pow(2) - self.gamma_logvar.exp())
            self.kld_loss = self.kld_weight * (alpha_KLD + gamma_KLD)

        num_examples_per_model = int(x.size(0) / self.num_members)
        extra = x.size(0) - (num_examples_per_model * self.num_members)


        # Repeated pattern: [[A,A],[B,B],[C,C]]
        alpha = torch.cat(
            [alpha_sample for i in range(num_examples_per_model)],
            dim=1).view([-1, self.in_channels])
        gamma = torch.cat(
            [gamma_sample for i in range(num_examples_per_model)],
            dim=1).view([-1, self.out_channels])
        if self.bias is not None:
            bias = torch.cat(
                [self.bias for i in range(num_examples_per_model)],
                dim=1).view([-1, self.out_channels])
            

        alpha.unsqueeze_(-1).unsqueeze_(-1).unsqueeze_(-1)
        gamma.unsqueeze_(-1).unsqueeze_(-1).unsqueeze_(-1)
        bias.unsqueeze_(-1).unsqueeze_(-1).unsqueeze_(-1)

        if extra != 0:
            alpha = torch.cat([alpha, alpha[:extra]], dim=0)
            gamma = torch.cat([gamma, gamma[:extra]], dim=0)
            if self.bias is not None:
                bias = torch.cat([bias, bias[:extra]], dim=0)

        result = self.conv(x*alpha)*gamma

        return result + bias if self.bias is not None else result

class Rank1_BNN_Conv3d(Rank_One_BNN_Conv3d):
    def __init__(self, in_channels, out_channels, num_members=4, **kwargs):
        super(Rank1_BNN_Conv3d, self).__init__(
            in_channels,
            out_channels,
            num_members=num_members,
            transpose=False,
            **kwargs)


class Rank1_BNN_ConvTranspose3d(Rank_One_BNN_Conv3d):
    def __init__(self, in_channels, out_channels, num_members=4, **kwargs):
        super(Rank1_BNN_ConvTranspose3d, self).__init__(
            in_channels,
            out_channels,
            num_members=num_members,
            transpose=True,
            **kwargs)


"""
LP BNN conv 3d
"""
class Latent_Posterior_BNN_Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, num_members=4, transpose=False, **kwargs):
        super(Latent_Posterior_BNN_Conv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_members = num_members 
        if transpose:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels, **kwargs)
        else:
            self.conv = nn.Conv3d(in_channels, out_channels, **kwargs)

        # TODO: pass these as args rather than had set
        self.random_sign_init = True
        self.constant_init = False
        self.probability = 0.5
        self.hidden_size=32
        self.latent_loss = 0

        
        self.alpha = nn.Parameter(torch.Tensor(num_members, in_channels))
        self.gamma = nn.Parameter(torch.Tensor(num_members, out_channels))
        if kwargs['bias']:
            self.bias = nn.Parameter(torch.Tensor(self.num_members, out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.first_layer = True if in_channels == 1 else False

        # VAE
        self.encoder_fc1 = nn.Linear(in_channels, self.hidden_size)
        self.encoder_fcmean = nn.Linear(self.hidden_size, self.hidden_size)
        self.encoder_fcvar = nn.Linear(self.hidden_size, self.hidden_size)
        self.decoder_fc1 = nn.Linear(self.hidden_size, in_channels)
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def reset_parameters(self):
        if self.constant_init:
            nn.init.constant_(self.alpha, 1.)
            if self.random_sign_init:
                if self.probability  == -1:
                    with torch.no_grad():
                        factor = torch.ones(
                            self.num_members, device=self.alpha_mean.device).bernoulli_(0.5)
                        factor.mul_(2).add_(-1)
                        self.alpha_mean.data = (self.alpha_mean.t() * factor).t()
                        if self.train_gamma:
                            self.gamma_mean.fill_(1.)
                            self.gamma_mean.data = (self.gamma_mean.t() * factor).t()
                elif self.probability  == -2:
                    with torch.no_grad():
                        positives_num = self.num_members // 2
                        factor1 = torch.Tensor([1 for i in range(positives_num)])
                        factor2 = torch.Tensor(
                            [-1 for i in range(self.num_members-positives_num)])
                        factor = torch.cat([factor1, factor2]).to(self.alpha_mean.device)
                        self.alpha_mean.data = (self.alpha_mean.t() * factor).t()
                        if self.train_gamma:
                            self.gamma_mean.fill_(1.)
                            self.gamma_mean.data = (self.gamma_mean.t() * factor).t()
                else:
                    with torch.no_grad():
                        self.alpha_mean.bernoulli_(self.probability)
                        self.alpha_mean.mul_(2).add_(-1)
                        if self.train_gamma:
                            self.gamma_mean.bernoulli_(self.probability)
                            self.gamma_mean.mul_(2).add_(-1)
        else:
            nn.init.normal_(self.alpha, mean=1., std=0.5)
            #nn.init.normal_(self.alpha, mean=1., std=1)
            nn.init.normal_(self.gamma, mean=1., std=0.5)
            #nn.init.normal_(self.gamma, mean=1., std=1)
            if self.random_sign_init:
                with torch.no_grad():
                    alpha_coeff = torch.randint_like(self.alpha, low=0, high=2)
                    alpha_coeff.mul_(2).add_(-1)
                    self.alpha *= alpha_coeff
                    gamma_coeff = torch.randint_like(self.gamma, low=0, high=2)
                    gamma_coeff.mul_(2).add_(-1)
                    self.gamma *= gamma_coeff
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def update_indices(self, indices):
        self.indices = indices

    def forward(self, x):
        # Train VAE
        embedded=F.relu(self.encoder_fc1(self.alpha))
        embedded_mean, embedded_logvar=self.encoder_fcmean(embedded),self.encoder_fcvar(embedded)
        z_embedded = self.reparameterize(embedded_mean, embedded_logvar)
        alpha_decoded = self.decoder_fc1(z_embedded)
        
        if self.training:
            MSE = F.mse_loss(alpha_decoded,self.alpha) #F.binary_cross_entropy(alpha_decoded, self.alpha, reduction='sum')
            KLD = -0.5 * torch.sum(1 + embedded_logvar - embedded_mean.pow(2) - embedded_logvar.exp())
            self.latent_loss = MSE + KLD

        num_examples_per_model = int(x.size(0) / self.num_members)
        extra = x.size(0) - (num_examples_per_model * self.num_members)


        # Repeated pattern: [[A,A],[B,B],[C,C]]
        alpha = torch.cat(
            [alpha_decoded for i in range(num_examples_per_model)],
            dim=1).view([-1, self.in_channels])
        gamma = torch.cat(
            [self.gamma for i in range(num_examples_per_model)],
            dim=1).view([-1, self.out_channels])
        if self.bias is not None:
            bias = torch.cat(
                [self.bias for i in range(num_examples_per_model)],
                dim=1).view([-1, self.out_channels])
            

        alpha.unsqueeze_(-1).unsqueeze_(-1).unsqueeze_(-1)
        gamma.unsqueeze_(-1).unsqueeze_(-1).unsqueeze_(-1)
        bias.unsqueeze_(-1).unsqueeze_(-1).unsqueeze_(-1)

        if extra != 0:
            alpha = torch.cat([alpha, alpha[:extra]], dim=0)
            gamma = torch.cat([gamma, gamma[:extra]], dim=0)
            if self.bias is not None:
                bias = torch.cat([bias, bias[:extra]], dim=0)

        result = self.conv(x*alpha)*gamma

        return result + bias if self.bias is not None else result

class LP_BNN_Conv3d(Latent_Posterior_BNN_Conv3d):
    def __init__(self, in_channels, out_channels, num_members=4, **kwargs):
        super(LP_BNN_Conv3d, self).__init__(
            in_channels,
            out_channels,
            num_members=num_members,
            transpose=False,
            **kwargs)


class LP_BNN_ConvTranspose3d(Latent_Posterior_BNN_Conv3d):
    def __init__(self, in_channels, out_channels, num_members=4, **kwargs):
        super(LP_BNN_ConvTranspose3d, self).__init__(
            in_channels,
            out_channels,
            num_members=num_members,
            transpose=True,
            **kwargs)




######################### Concrete Dropout ##################################
'''
https://github.com/aurelio-amerio/ConcreteDropout/
'''
class ConcreteDropout(nn.Module):
    def __init__(self, layer,
                 weight_regularizer=1e-6,
                 dropout_regularizer=1e-5,
                 init_min=0.1,
                 init_max=0.1,
                 is_mc_dropout=False,
                 temperature=0.1):

        super().__init__()
        self.layer = layer
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer

        init_min = np.log(init_min) - np.log(1.0 - init_min)
        init_max = np.log(init_max) - np.log(1.0 - init_max)

        self.p_logit = nn.Parameter(
            torch.empty(1).uniform_(init_min, init_max))
        self.p = torch.sigmoid(self.p_logit)
        self.temperature = temperature
        self.is_mc_dropout = is_mc_dropout

        self.regularization = 0

    def _get_noise_shape(self, x):
        raise NotImplementedError(
            "Subclasses of ConcreteDropout must implement the noise shape")

    def _concrete_dropout(self, x):
        '''
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        '''
        p = self.p
        # machine precision epsilon for numerical stability inside the log
        eps = torch.finfo(x.dtype).eps

        # this is the shape of the dropout noise, same as tf.nn.dropout
        noise_shape = self._get_noise_shape(x)

        unif_noise = torch.rand(*noise_shape).to(x.device)  # uniform noise
        # bracket inside equation 5, where u=uniform_noise
        drop_prob = (
            torch.log(p + eps)
            - torch.log1p(eps - p)
            + torch.log(unif_noise + eps)
            - torch.log1p(eps - unif_noise)
        )
        drop_prob = torch.sigmoid(drop_prob / self.temperature)  # z of eq 5
        random_tensor = 1. - drop_prob

        retain_prob = 1. - p
        x = x + random_tensor  # we multiply the input by the concrete dropout mask
        x = x/retain_prob  # we normalise by the probability to retain

        return x

    def get_regularization(self, x, layer):
        p = self.p
        # We will now compute the KL terms following eq.3 of 1705.07832
        weight = layer.weight
        # The kernel regularizer corresponds to the first term
        # Note: we  divide by (1 - p) because  we  scaled  layer  output  by(1 - p)
        kernel_regularizer = self.weight_regularizer * torch.sum(torch.square(
            weight)) / (1. - p)
        # the dropout regularizer corresponds to the second term
        dropout_regularizer = p * torch.log(p)
        dropout_regularizer += (1. - p) * torch.log1p(- p)
        dropout_regularizer *= self.dropout_regularizer * x.shape[1]
        # this is the KL term to be added as a loss
        # regularizer
        return torch.sum(kernel_regularizer + dropout_regularizer)

    def forward(self, x):
        self.p = torch.sigmoid(self.p_logit)
        self.regularization = self.get_regularization(x, self.layer)
        x = self._concrete_dropout(x)
        return self.layer(x)

'''
https://github.com/aurelio-amerio/ConcreteDropout/
'''
class ConcreteLinearDropout(ConcreteDropout):
    """
    This wrapper allows to learn the dropout probability for any given Linear input layer.
    ```python
        x = # some input layer
        cd = ConcreteLinearDropout()
        linear = torch.Linear(in_features, out_features)
        x = cd(x, linear)
    ```
    # Arguments
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
        temperature:
            The temperature of the Concrete Distribution. 
            Must be low enough in order to have the correct KL approximation.
            For:
                $t \rightarrow 0$ we obtain the discrete distribution
                $t \rightarrow \infty$ we obtain a uniform distribution
    """

    def __init__(self, temperature=0.1, **kwargs):
        super(ConcreteLinearDropout, self).__init__(
            temperature=temperature, **kwargs)

    # implement the noise shape for regular dropout
    def _get_noise_shape(self, inputs):
        input_shape = inputs.shape
        return input_shape

'''
https://github.com/aurelio-amerio/ConcreteDropout/
'''
class ConcreteDropout1D(ConcreteDropout):
    """
    This wrapper allows to learn the dropout probability for any given Conv1d input layer. 
    It is the Concrete Dropout implementation of `Dropout1d`.

    ```python
        x = # some input layer
        cd = ConcreteDropout1D()
        conv1d = torch.Conv1d(in_channels,out_channels, kernel_size)
        x = cd(x, conv1d)
    ```
    # Arguments
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
        temperature:
            The temperature of the Concrete Distribution. 
            Must be low enough in order to have the correct KL approximation.
            For:
                $t \rightarrow 0$ we obtain the discrete distribution
                $t \rightarrow \infty$ we obtain a uniform distribution
    """

    def __init__(self, temperature=2./3., **kwargs):
        super(ConcreteDropout1D, self).__init__(
            temperature=temperature, **kwargs)

    # implement the noise shape for spatial dropout
    def _get_noise_shape(self, inputs):
        input_shape = inputs.shape
        return (input_shape[0], input_shape[1], 1)

'''
https://github.com/aurelio-amerio/ConcreteDropout/
'''
class ConcreteDropout2D(ConcreteDropout):
    """
    This wrapper allows to learn the dropout probability for any given Conv2d input layer. 
    It is the Concrete Dropout implementation of `Dropout2d`.

    ```python
        x = # some input layer
        cd = ConcreteDropout2D()
        conv2d = torch.Conv2d(in_channels,out_channels, kernel_size)
        x = cd(x, conv2d)
    ```
    # Arguments
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
        temperature:
            The temperature of the Concrete Distribution. 
            Must be low enough in order to have the correct KL approximation.
            For:
                $t \rightarrow 0$ we obtain the discrete distribution
                $t \rightarrow \infty$ we obtain a uniform distribution
    """

    def __init__(self, temperature=2./3., **kwargs):
        super(ConcreteDropout2D, self).__init__(
            temperature=temperature, **kwargs)

    # implement the noise shape for spatial dropout
    def _get_noise_shape(self, inputs):
        input_shape = inputs.shape
        return (input_shape[0], input_shape[1], 1, 1)

'''
https://github.com/aurelio-amerio/ConcreteDropout/
'''
class ConcreteDropout3D(ConcreteDropout):
    """
    This wrapper allows to learn the dropout probability for any given Conv3d input layer. 
    It is the Concrete Dropout implementation of `Dropout3d`.

    ```python
        x = # some input layer
        cd = ConcreteDropout3D()
        conv3d = torch.Conv3d(in_channels,out_channels, kernel_size)
        x = cd(x, conv3d)
    ```
    # Arguments
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
        temperature:
            The temperature of the Concrete Distribution. 
            Must be low enough in order to have the correct KL approximation
            For:
                $t \rightarrow 0$ we obtain the discrete distribution
                $t \rightarrow \infty$ we obtain a uniform distribution
    """

    def __init__(self, layer, temperature=2./3., **kwargs):
        super(ConcreteDropout3D, self).__init__(
            layer, temperature=temperature, **kwargs)

    # implement the noise shape for spatial dropout
    def _get_noise_shape(self, inputs):
        input_shape = inputs.shape
        return (input_shape[0], input_shape[1], 1, 1, 1)

'''
https://github.com/aurelio-amerio/ConcreteDropout/
'''
def get_weight_regularizer(N, l=1e-2, tau=0.1):
    return l**2 / (tau * N)

'''
https://github.com/aurelio-amerio/ConcreteDropout/
'''
def get_dropout_regularizer(N, tau=0.1, cross_entropy_loss=False):
    reg = 1 / (tau * N)
    if not cross_entropy_loss:
        reg *= 2
    return reg