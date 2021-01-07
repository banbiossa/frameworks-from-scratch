from frameworks_from_scratch import *


def test_function():
    x = Variable(np.array(2))
    y = Variable(np.array(3))
    z = add(square(x), square(y))
    z.backward()

    assert x.grad == 4.0
    assert y.grad == 6.0
