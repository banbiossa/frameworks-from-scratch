import logging
import numpy as np

logger = logging.getLogger(__name__)


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"only np arrays, not {type(data)}")
        self.data = data
        self.grad = None
        self.creator = None

    def __repr__(self):
        return f"Variable({self.data:.2f})"

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            logger.info(f"Gradient from {f} and {self}")
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                x.grad = gx

                if x.creator is not None:
                    funcs.append(x.creator)
