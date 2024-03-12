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

    c.grad = np.array(1.0)
    c.backward()
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

    x = Variable.Variable(np.array(2.0))
    y = Variable.Variable(np.array(3.0))

    z = add(square(x), square(y))
    z.backward()
    print(x.grad, y.grad, z.data)

    x = Variable.Variable(np.array(3.0))
    y = add(x, x)
    z = add(y, x)
    z.backward()
    print(x.grad, y.data, z.data)

    x.cleargrad()
    y.backward()
    print(x.grad, y.data)

    x = Variable.Variable(np.array(2.0))
    a = square(x)
    y = add(square(a), square(a))
    y.backward()

    print(x.grad, y.data)

    for i in range(10):
        x = Variable.Variable(np.random.randn(10000))
        y = square(square(square(x)))
        print(y.data)
    
    x = Variable.Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    print(x.shape, x.ndim, x.size, x.dtype, len(x))
    print(x)

    a = Variable.Variable(np.array(3.0))
    b = Variable.Variable(np.array(2.0))
    c = Variable.Variable(np.array(1.0))
    y = a * b + c
    y.backward()
    print(y)
    print(a.grad, b.grad, c.grad)