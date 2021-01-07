import numpy as np
from .variable import Variable
import logging

logger = logging.getLogger(__name__)


class Function:
    def __call__(self, input_var: Variable) -> Variable:
        x = input_var.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.input = input_var
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.input.__repr__()})"

    def __str__(self):
        return f"[{self.__repr__()} -> {self.output.__repr__()}]"


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        logger.info(f"Backward called: {gx=:.2f} := 2 * {x=:.2f} * {gy=:.2f}")
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        logger.info(f"Backward called: {gx=:.2f} := {np.exp(x)=:.2f} * {gy=:.2f}")
        return gx
