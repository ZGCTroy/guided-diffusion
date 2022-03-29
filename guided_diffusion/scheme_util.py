from tensorflow.python.client.session import _get_attrs_values
import torch as th
from guided_diffusion import dist_util

import torch.nn as nn
import torch as th


def VarianceMSE(predict_variance, timesteps, diffusion,
                loss_type='l1',
                diffusion_variance_attr='alphas_cumprod'):
    # var
    if loss_type == 'mse':
        loss_fn = nn.MSELoss(reduce=False, size_average=True)
    elif loss_type == 'l1':
        loss_fn = nn.L1Loss(reduce=False, size_average=True)

    # mind the loss log dimension
    diffusion_variance = th.FloatTensor(1.0 - getattr(diffusion, diffusion_variance_attr)).to(dist_util.dev())
    # reverse the scale
    diffusion_variance = diffusion_variance[timesteps]
    # print(variance)
    return loss_fn(predict_variance, diffusion_variance)

