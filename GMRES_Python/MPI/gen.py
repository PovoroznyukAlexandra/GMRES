import numpy as np
import sys 

args = sys.argv

n = int(args[1])
A = np.random.rand(n, n)
for i in range(n):
    A[i, i] = 2 * A[i, :].sum()
x_sol = np.random.rand(n)
b = A @ x_sol

A.tofile("A.npy")
b.tofile("b.npy")

