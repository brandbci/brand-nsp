import numpy as np
from numba import njit, prange
import time

# Type specialized implementations
@njit('float64[:,:](float64[:,:], float64[:,:])')
def numba_dot(A, B):
    return np.dot(A, B)

# @njit('float64[:,:](float64[:,:], float64[:,:], int32)')
# def numba_block_dot(A, B, block_size):
#     m, n = A.shape[0], B.shape[1]
#     k = A.shape[1]
#     result = np.zeros((m, n))
    
#     for i in range(0, m, block_size):
#         for j in range(0, n, block_size):
#             for k_block in range(0, k, block_size):
#                 i_end = min(i + block_size, m)
#                 j_end = min(j + block_size, n)
#                 k_end = min(k_block + block_size, k)
                
#                 for ii in range(i, i_end):
#                     for jj in range(j, j_end):
#                         temp = 0.0
#                         for kk in range(k_block, k_end):
#                             temp += A[ii, kk] * B[kk, jj]
#                         result[ii, jj] += temp
#     return result


@njit('float64[:,:](float64[:,:], float64[:,:], int32)',parallel = True)
def numba_block_dot(A, B, block_size):
    m, k = A.shape
    result = np.zeros((m, B.shape[1]))
    n = 0
    
    while n < m:
        end = min(n + block_size, m)
        result[n:end, :] = np.dot(A[n:end, :], B)
        n += block_size
        
    return result



@njit('float64[:,:](float64[:,:], float64[:,:])', parallel=True)
def numba_parallel_dot(A, B):
    m, n = A.shape[0], B.shape[1]
    result = np.zeros((m, n))
    
    for i in prange(m):
        for j in range(n):
            temp = 0.0
            for k in range(A.shape[1]):
                temp += A[i, k] * B[k, j]
            result[i, j] = temp
    return result

def numpy_block_dot(A, B, block_size):
    m, k = A.shape
    result = np.zeros((m, B.shape[1]))
    n = 0
    
    while n < m:
        end = min(n + block_size, m)
        result[n:end, :] = np.dot(A[n:end, :], B)
        n += block_size
        
    return result

# Create test matrices
A = np.random.random((256, 256))
B = np.random.random((256, 30))
BLOCK_SIZE = 16

# Warm up JIT
_ = numba_dot(A, B)
_ = numba_block_dot(A, B, BLOCK_SIZE)
_ = numba_parallel_dot(A, B)

methods = {
    'NumPy dot': lambda: np.dot(A, B),
    'NumPy block dot': lambda: numpy_block_dot(A, B, BLOCK_SIZE),
    'Numba dot': lambda: numba_dot(A, B),
    'Numba block dot': lambda: numba_block_dot(A, B, BLOCK_SIZE),
    'Numba parallel dot': lambda: numba_parallel_dot(A, B)
}

results = {}
times = {}

# Run benchmarks
for name, func in methods.items():
    start = time.time()
    for _ in range(1000):  # Run 100 times for more stable measurements
        results[name] = func()
    times[name] = (time.time() - start) / 1000  # Average time

# Print results
for name, t in times.items():
    print(f"{name}: {t*1000:.3f} ms")

# # Verify all results match
# base_result = results['NumPy dot']
# for name, result in results.items():
#     if name != 'NumPy dot':
#         print(f"\n{name} matches NumPy:", np.allclose(result, base_result))