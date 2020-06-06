import logging

logger = logging.getLogger(__name__)


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def __repr__(self):
        return f"Variable({self.data:.2f})"

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator
        if f is not None:
            logger.info(f"Gradient from {f} and {self}")
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward()
