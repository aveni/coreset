# Adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py
# Adapted from https://github.com/kohpangwei/influence-release/influence/dataset.py
import numpy as np

class DataSet(object):

    def __init__(self, x, y, ix=None):

        assert(x.shape[0] == y.shape[0])

        x = x.astype(np.float32)
        y = y.astype(np.float32)

        if ix is not None:
            self._ix = np.array(ix)
        else:
            self._ix = np.arange(x.shape[0])

        self._x = x
        self._y = y
        self._index_in_epoch = 0

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def size(self):
        return len(self._ix)

    def set_ix(self, ix):
        if ix is not None:
            self._ix = np.array(ix)
        else:
            self._ix = np.arange(self._x.shape[0])
        self.reset_batch()

    def reset_batch(self):
        self._index_in_epoch = 0        

    def next_batch(self, batch_size):
        assert batch_size <= len(self._ix)

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > len(self._ix):

            # Shuffle the data
            perm = np.arange(len(self._ix))
            np.random.shuffle(perm)
            self._ix = self._ix[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size

        end = self._index_in_epoch
        return self._x[self._ix[start:end]], self._y[self._ix[start:end]], self._ix[start:end]