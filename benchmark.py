import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, pmap
import time

# Configurações do teste
matrix_size = 10000
iterations = 10

# Função para benchmark
def benchmark(func, *args, iterations=10):
    times = []
    for _ in range(iterations):
        start = time.time()
        func(*args)
        end = time.time()
        times.append(end - start)
    avg_time = sum(times) / len(times)
    return avg_time

# NumPy Benchmark
def numpy_operations():
    a = np.random.rand(matrix_size, matrix_size)
    b = np.random.rand(matrix_size, matrix_size)
    c = a + b  # Soma
    d = np.dot(a, b)  # Produto matricial
    e = np.exp(c)  # Exponencial
    return e

# JAX Benchmark
def jax_operations():
    a = jnp.array(np.random.rand(matrix_size, matrix_size))
    b = jnp.array(np.random.rand(matrix_size, matrix_size))
    c = a + b  # Soma
    d = jnp.dot(a, b)  # Produto matricial
    e = jnp.exp(c)  # Exponencial
    return e

# Benchmark: NumPy
print("Benchmark: NumPy")
numpy_time = benchmark(numpy_operations, iterations=iterations)

# # Benchmark: JAX (sem JIT)
print("Benchmark: JAX (sem JIT)")
jax_time = benchmark(jax_operations, iterations=iterations)

# Benchmark: JAX (com JIT)
print("Benchmark: JAX (com JIT)")
jax_operations_jit = jit(jax_operations)
jax_jit_time = benchmark(jax_operations_jit, iterations=iterations)

# Resultados
print(f"NumPy Time: {numpy_time:.6f} s")
print(f"JAX Time (no JIT): {jax_time:.6f} s")
print(f"JAX Time (with JIT): {jax_jit_time:.6f} s")


