import numpy as np
from .variable import Variable
from .util import as_array
import logging
from typing import List
import weakref
from .config import Config

logger = logging.getLogger(__name__)


class Function:
    def __call__(self, *inputs: Variable) -> List[Variable]:
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)

            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x0, x1):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.inputs.__repr__()})"

    def __str__(self):
        return f"[{self.__repr__()} -> {self.outputs.__repr__()}]"


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


def mul(x0, x1):
    return Mul()(x0, x1)


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        logger.info(f"Backward called: {gx=:.2f} := 2 * {x=:.2f} * {gy=:.2f}")
        return gx


def square(x):
    return Square()(x)


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        logger.info(f"Backward called: {gx=:.2f} := {np.exp(x)=:.2f} * {gy=:.2f}")
        return gx


def exp(x):
    return Exp()(x)


class Log(Function):
    def forward(self, x):
        return np.log(x)

    def backward(self, gy):
        x = self.input.data
        gx = gy / x
        return gx
