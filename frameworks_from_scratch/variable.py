import logging
import numpy as np
import frameworks_from_scratch  # to avoid circular import error

logger = logging.getLogger(__name__)


class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                data = np.array(data)
        self.data = data
        self.grad = None
        self.creator = None
        self.name = name
        self.generation = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __mul__(self, other):
        return frameworks_from_scratch.mul(self, other)

    def __rmul__(self, other):
        return frameworks_from_scratch.mul(self, other)

    def __add__(self, other):
        return frameworks_from_scratch.add(self, other)

    def __radd__(self, other):
        return frameworks_from_scratch.add(self, other)

    def __neg__(self):
        return frameworks_from_scratch.neg(self)

    def __sub__(self, other):
        return frameworks_from_scratch.sub(self, other)

    def __rsub__(self, other):
        return frameworks_from_scratch.sub(other, self)

    def __truediv__(self, other):
        return frameworks_from_scratch.div(self, other)

    def __pow__(self, power, modulo=None):
        return frameworks_from_scratch.pow(self, power)

    def __rtruediv__(self, other):
        return frameworks_from_scratch.div(other, self)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return "Variable(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return f"Variable({p})"

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            logger.info(f"Gradient from {f} and {self}")
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx  # don't use `+=`!

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None  # y is a weakref
