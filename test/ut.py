import unittest
from src.Variable import *
from src.Function import *
from src.Operator import *

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable.Variable(x.data - eps)
    x1 = Variable.Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)
    

class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable.Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable.Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable.Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        self.assertTrue(np.allclose(x.grad, num_grad))

class MultiTest(unittest.TestCase):
    def test_forward(self):
        x = Variable.Variable(np.array([[1.0, 2.0], [3.0, 4.0]]))
        y = square(x)
        expected = np.array([[1.0, 4.0], [9.0, 16.0]])
        self.assertTrue(np.allclose(y.data, expected))

    def test_backward(self):
        x = Variable.Variable(np.array([[1.0, 2.0], [3.0, 4.0]]))
        y = square(x)
        y.backward()
        expected = np.array([[2.0, 4.0], [6.0, 8.0]])
        self.assertTrue(np.allclose(x.grad, expected))

    def test_gradient_check(self):
        x = Variable.Variable(np.random.rand(2, 2))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        self.assertTrue(np.allclose(x.grad, num_grad))

class AddTest(unittest.TestCase):
    def test_forward(self):
        x = Variable.Variable(np.array(2.0))
        y = Variable.Variable(np.array(3.0))
        z = add(x, y)
        expected = np.array(5.0)
        self.assertEqual(z.data, expected)

    def test_backward(self):
        x = Variable.Variable(np.array(3.0))
        y = Variable.Variable(np.array(2.0))
        z = add(x, y)
        z.backward()
        expected = np.array(1.0)
        self.assertEqual(x.grad, expected)
        self.assertEqual(y.grad, expected)