from jax import numpy as jnp
import jax
import matplotlib.pyplot as plt
from typing import Callable


def visualize_2d_function(
  f: Callable,
  x: jnp.ndarray = jnp.linspace(-5, 5, 100),
  y: jnp.ndarray = jnp.linspace(-5, 5, 100),
  levels: jnp.ndarray = jnp.logspace(0, 5, 35)
):
  X, Y = jnp.meshgrid(x, y)
  Z = jax.vmap(lambda x, y: f(jnp.array([x, y])))(X, Y)

  plt.figure(figsize=(10, 8))
  plt.contour(X, Y, Z, levels=levels)
  plt.colorbar(label='f(x, y)')
  plt.xlabel("x")
  plt.ylabel("y")
  

def visualize_opt_trajectory(
  f: Callable,
  trajectory: jnp.ndarray,
  x: jnp.ndarray = jnp.linspace(-5, 5, 100),
  y: jnp.ndarray = jnp.linspace(-5, 5, 100),
  levels: jnp.ndarray = jnp.logspace(0, 5, 35)
):
  visualize_2d_function(f, x, y, levels)
  trajectory = jnp.array(trajectory)
  plt.scatter(
    trajectory[:, 0], 
    trajectory[:, 1], 
    c=jnp.arange(len(trajectory)), 
    cmap='viridis', 
    label='Optimization trajectory'
  )
  plt.show()
  