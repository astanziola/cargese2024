import warnings
import traceback
from typing import Any, Callable, Tuple
from termcolor import colored
from IPython.display import HTML, display

import jax
import jax.numpy as jnp

# Constants
COLORS = {
    'red': "#FFCDD2",
    'green': "#C8E6C9",
    'blue': "#e4f8fb",
    'yellow': "#ffe0b2"
}

def tuples_to_string(tuples: Tuple) -> str:
  """Converts tuples to string, with each tuple on a new line."""
  return "\n".join([str(t) for t in tuples])

def colorize(text: str, color: str) -> str:
    """
    Colorize text using termcolor.

    Args:
        text (str): The text to colorize.
        color (str): The color to use.

    Returns:
        str: Colorized text.
    """
    return colored(text, color)

def report(passed: bool, expected: Any, received: Any) -> None:
    """
    Report the test result.

    Args:
        passed (bool): Whether the test passed.
        expected (Any): The expected result.
        received (Any): The actual result.
    """
    result = "PASSED" if passed else "FAILED"
    color = "green" if passed else "red"
    print(colorize(f"Test {result} \n", color))
    print(colorize(f"Expecting:\n{tuples_to_string(expected)}\n", color))
    print(colorize(f"Received:\n{tuples_to_string(received)}\n", color))

def verify(func: Callable) -> Callable:
    """
    Decorator to verify a function's output.

    Args:
        func (Callable): The function to verify.

    Returns:
        Callable: Wrapped function.
    """
    def wrapper(*args, **kwargs) -> None:
        print(colorize("Verifying...", "yellow"))
        try:
            passed, expected, received = func(*args, **kwargs)
            report(passed, expected, received)
        except Exception as e:
            report(False, None, traceback.format_exc())

    return wrapper

@verify
def verify_test(test_func: Callable) -> Tuple[bool, float, float]:
    result = test_func()
    expected = 1.0
    passed = result == expected
    return passed, expected, result
  
  
###########################################################################
#                                 Basic jax                               #
###########################################################################

@verify
def verify_create_arrays(fun) -> Tuple[bool, Tuple[float, float], Tuple[float, float]]:
  x_hat = jnp.arange(2, 21, 2)
  y_hat = jnp.arange(1, 20, 2)
  
  x, y = fun()
  
  try:
    passed = jnp.allclose(x, x_hat) and jnp.allclose(y, y_hat)
  except Exception as e:
    print(e)
  
  return passed, (x_hat, y_hat), (x, y)
  
  
@verify
def verify_matrix_exercise(fun):
  matrix_square_hat = jnp.array([[30, 36, 42], [66, 81, 96], [102, 126, 150]])
  matrix_elementwise_squared_hat = jnp.array([[1, 4, 9], [16, 25, 36], [49, 64, 81]])
  
  matrix_square, matrix_elementwise_squared = fun()
  
  try:
    passed = jnp.allclose(matrix_square, matrix_square_hat) and jnp.allclose(matrix_elementwise_squared, matrix_elementwise_squared_hat)
  except Exception as e:
    print(e)
  
  return passed, (matrix_square_hat, matrix_elementwise_squared_hat), (matrix_square, matrix_elementwise_squared)


@verify
def verify_array_2d_exercise(fun):
  _matrix = jnp.arange(1,11)
  _matrix = _matrix.reshape(5,2)
  _matrix_modified = _matrix.at[::2].set(-1)
  
  matrix, matrix_modified = fun()
  
  passed = False
  
  try:
    passed = jnp.allclose(matrix, _matrix) and jnp.allclose(matrix_modified, _matrix_modified)
  except Exception as e:
    print(e)
  
  return passed, (_matrix, _matrix_modified), (matrix, matrix_modified)
  
  
@verify
def verify_mse(fun):
  y_hat = jnp.array([1, 2, 3, 4, 5])
  y = jnp.array([1, 3, 3, 3, 5])
  
  mse_hat = 0.4
  
  mse = fun(y_hat, y)
  
  try:
    passed = jnp.allclose(mse, mse_hat)
  except Exception as e:
    print(e)
  
  return passed, (mse_hat,), (mse,)


@verify
def verify_rosenbrock(point, value):
  def f(p):
    x, y = p
    return (1-x)**2 + 100*(y-x**2)**2
  
  gradient_fun = jax.jacrev(f)
  _gradient = gradient_fun(point)
  
  try:
    passed = jnp.allclose(_gradient, value) and _gradient.shape == value.shape
  except Exception as e:
    print(e)
  
  return passed, (_gradient,), (value,)


@verify
def verify_sphere_to_cart(J):
  _J = jnp.asarray([[0.46452138,0.8503006,-0.11861178], [0.11861178,0.2171174,0.46452138], [0.87758255,-0.47942555,0.]])
  
  try:
    passed = jnp.allclose(J, _J)
  except Exception as e:
    print(e)
    
  return passed, (_J,), (J,)


@verify
def verify_rosenbrock_optimization(minimum):
  x, y = minimum
  val = None
  
  try:
    val = (1-x)**2 + 100*(y-x**2)**2
    passed = val < 1e-3
  except Exception as e:
    print(e)
    
  return passed, ("Value of function at minimum < 0.001",), (f"Value of function at minimum: {val}",)