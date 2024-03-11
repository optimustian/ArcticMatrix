import numpy as np
from src.Variable import *
from src.Function import *
from src.Operator import *

def g(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))

if __name__ == "__main__":
    x = np.random.randn(10, 2)

    var = Variable.Variable(x)
    print(var.data)

    f = Square()
    y = f(var)
    z = Exp()(y)
    print(z.data)

    A = Square()
    B = Exp()
    C = Square()

    x = Variable.Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    c = C(b)

    y.grad = np.array(1.0)
    b.grad = C.backward(y.grad)
    a.grad = B.backward(b.grad)
    x.grad = A.backward(a.grad)
    print(x.grad)

    assert c.creator == C
    assert c.creator.input == b
    assert c.creator.input.creator == B
    assert c.creator.input.creator.input == a
    assert c.creator.input.creator.input.creator == A
    assert c.creator.input.creator.input.creator.input == x

    c.grad = np.array(1.0)
    c.backward()
    print(x.grad)

    c.grad = np.array(1.0)
    c.backward_recursion()
    print(x.grad)

    x = Variable.Variable(np.array(0.5))
    a = square(x)
    b = exp(a)
    y = square(b)

    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)

    x = Variable.Variable(np.array(0.5))
    y = square(exp(square(x)))
    y.backward()
    print(x.grad)