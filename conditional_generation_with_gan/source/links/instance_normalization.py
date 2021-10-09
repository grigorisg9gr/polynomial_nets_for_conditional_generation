import numpy

from chainer import configuration
from chainer import cuda
from chainer.functions.normalization import batch_normalization
from chainer import initializers
from chainer import link
from chainer.utils import argument
from chainer import variable
from chainer import functions as F


class InstanceNormalization(link.Link):
    def __init__(self, size, eps=2e-5, dtype=numpy.float32,
                 use_gamma=True, use_beta=True,
                 initial_gamma=None, initial_beta=None):
        super(InstanceNormalization, self).__init__()
        self.eps = eps

        with self.init_scope():
            if use_gamma:
                if initial_gamma is None:
                    initial_gamma = 1
                initial_gamma = initializers._get_initializer(initial_gamma)
                initial_gamma.dtype = dtype
                self.gamma = variable.Parameter(initial_gamma, size)
            if use_beta:
                if initial_beta is None:
                    initial_beta = 0
                initial_beta = initializers._get_initializer(initial_beta)
                initial_beta.dtype = dtype
                self.beta = variable.Parameter(initial_beta, size)

    def __call__(self, x, **kwargs):
        """__call__(self, x, finetune=False)
        Invokes the forward propagation of InstanceNormalization.

        Args:
            x (Variable): Input variable.
        """
        argument.check_unexpected_kwargs(
            kwargs, test='test argument is not supported anymore. '
                         'Use chainer.using_config')
        h = x
        if hasattr(self, 'gamma'):
            gamma = self.gamma
        else:
            with cuda.get_device_from_id(self._device_id):
                gamma = variable.Variable(self.xp.ones(
                    h.shape[1], dtype=h.dtype))
        if hasattr(self, 'beta'):
            beta = self.beta
        else:
            with cuda.get_device_from_id(self._device_id):
                beta = variable.Variable(self.xp.zeros(
                    h.shape[1], dtype=h.dtype))
        axes = tuple(list(range(1, h.ndim)))
        mean = F.broadcast_to(F.mean(h, axes, keepdims=True), h.shape)
        var = F.broadcast_to(F.mean(F.squared_difference(h, mean), axes, keepdims=True), h.shape)
        z = (h-mean) / (F.sqrt(var) + self.eps)
        if gamma.ndim != h.ndim:
            gamma = F.broadcast_to(F.reshape(gamma, [1] + list(gamma.shape) + [1] * (h.ndim - 2)), h.shape)
            beta = F.broadcast_to(F.reshape(beta, [1] + list(beta.shape) + [1] * (h.ndim - 2)), h.shape)
        return gamma * z + beta
