#!/usr/bin/env python
# coding=utf-8
from numba import jit
from numpy import arange
import time


@jit
def sum2d(arr):
    M, N = arr.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i, j]
    return result


a = arange(9).reshape(3, 3)
start_time = time.time()
for i in range(10000000):
    sum2d(a)
end_time = time.time()
print("numba time cost : ", end_time - start_time)
