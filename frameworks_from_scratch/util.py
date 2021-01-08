from .variable import Variable
import os
import subprocess
import numpy as np
from dotenv import find_dotenv
from pathlib import Path
import math


def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def reshape_sum_backward(gy, x_shape, axis, keepdims):
    """Reshape gradient appropriately for f.sum's backward.

    Args:
        gy (Variable): Gradient variable from the output by backprop.
        x_shape (tuple): Shape used at sum function's forward.
        axis (None or int or tuple of ints): Axis used at sum functions's forward.
        keepdims (bool): Keepdims used at sum function's forward.

    Returns:
        Variable: Gradient variable which is reshaped appropriately.
    """
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)
    return gy


def variable_to_dot(v, verbose=False):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = "" if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ": "
        name += str(v.shape) + " " + str(v.dtype)

    return dot_var.format(id(v), name)


def function_to_dot(f):
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = "{} -> {}\n"
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))  # y is weakref
    return txt


def graph_to_dot(output, verbose=True):
    txt = ""
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            # funcs.sort(key=lambda x: x.generation)
            seen_set.add(f)

    add_func(output.creator)
    txt += variable_to_dot(output, verbose)

    while funcs:
        func = funcs.pop()
        txt += function_to_dot(func)
        for x in func.inputs:
            txt += variable_to_dot(x, verbose)

            if x.creator is not None:
                add_func(x.creator)

    return "digraph g {\n" + txt + "}"


def plot_dot_graph(output, verbose=True, to_file="graph.png"):
    dot_graph = graph_to_dot(output, verbose)

    base_path = Path(find_dotenv()).parent
    extension = os.path.splitext(to_file)[1][1:]
    filename = to_file[: -len(extension) - 1]

    graph_path = Path(base_path, "dot", "input", f"{filename}.dot")
    output_path = Path(base_path, "dot", "output", to_file)

    with open(graph_path, "w") as f:
        f.write(dot_graph)

    cmd = "dot {} -T {} -o {}".format(graph_path, extension, output_path)
    subprocess.run(cmd, shell=True)


def goldstein(x, y):
    z = (
        1
        + (x + y + 1) ** 2
        * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)
    ) * (
        30
        + (2 * x - 3 * y) ** 2
        * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2)
    )
    return z


def taylor_sin(x, threshold=1e-4):
    y = 0

    for i in range(int(1e6)):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y


def rosenbrock(x0, x1):
    return 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
