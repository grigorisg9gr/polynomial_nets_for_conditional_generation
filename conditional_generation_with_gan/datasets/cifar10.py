import os

import numpy as np
from PIL import Image
import chainer
from chainer import cuda
from chainer.dataset import dataset_mixin
import scipy.misc

ORGSIZE = 32


class CIFAR10Dataset(dataset_mixin.DatasetMixin):
    def __init__(self, size=32, train=True, dequantize=True, resize_method='bilinear',
                 seed=9, n_samples=None, n_reduce=None, exclude_class=-100, **kwargs):
        data_train, data_test = chainer.datasets.get_cifar10(ndim=3, withlabel=True, scale=255)
        if n_reduce is not None and train:
            if isinstance(exclude_class, int):
                exclude_class = [exclude_class]
            data_train_reduced = []
            samples_each_class = np.zeros((10), dtype=np.int)
            for i in range(len(data_train)):
                # # find the class of the sample; by default the second attribute of the tuple.
                class_sample = data_train[i][1]
                # # if a) we have not filled the positions for the class, or b) the class is
                # #  in the 'exclude_class', then we insert the sample into the reduced list.
                if samples_each_class[class_sample] < n_reduce or (class_sample in exclude_class):
                    data_train_reduced.append(data_train[i])
                    samples_each_class[class_sample] += 1
            data_train = data_train_reduced
        self.data = data_train if train else data_test 
        if n_samples is not None:
            self.data = self.data[:n_samples]
        self.size = size
        self.resize_method = resize_method
        self.dequantize = dequantize

    def __len__(self):
        return len(self.data)

    def get_example(self, i):
        image = self.data[i][0]
        label = self.data[i][1]
        image = np.asarray(image, np.uint8)
        if self.size != ORGSIZE:
            image = scipy.misc.imresize(image.transpose(1, 2, 0),
                                        [self.size, self.size], self.resize_method).transpose(2, 0, 1)
        image = np.array(image / 128. - 1., np.float32)
        if self.dequantize:
            image += np.random.uniform(size=image.shape, low=0., high=1. / 128).astype(np.float32)

        return (image, label)
