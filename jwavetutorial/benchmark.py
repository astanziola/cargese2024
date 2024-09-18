import time
from typing import Callable, Iterable
from jax import numpy as jnp


def time_execution(func, *args):
    start_time = time.time()
    result = func(*args)
    result.block_until_ready()
    end_time = time.time()
    return result, end_time - start_time

def mc_pi_speed_comparison(
  func: Callable,
  jit_fun: Callable, 
  n_points_list: Iterable[int]
):
  print("Non-JIT vs JIT Monte Carlo Pi Estimation:")
  print("-----------------------------------------------------------")
  print(" n_points | Non-JIT Time | JIT Time | Speedup | Pi Estimate")
  print("-----------------------------------------------------------")
  
  for n_points in n_points_list:
    # Warm-up run for JIT (compilation occurs here)
    _ = jit_fun(n_points)
    
    # Non-JIT execution
    _, non_jit_time = time_execution(func, n_points)
    
    # JIT execution
    jit_result, jit_time = time_execution(jit_fun, n_points)
    
    # Calculate speedup
    speedup = non_jit_time / jit_time
    
    print(f"{n_points:9d} | {non_jit_time:.6f}s | {jit_time:.6f}s | {speedup:.2f}x | {jit_result:.6f}")

  print("-----------------------------------------------------------")
  print(f"True value of pi: {jnp.pi:.6f}")