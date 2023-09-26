import numpy as np
from mpi4py import MPI
import sys 
from numpy.linalg import norm as norm
import time
import cProfile


def ddot(x, y, comm):
    local_ddot = np.inner(x, y)
    global_ddot = comm.allreduce(local_ddot, MPI.SUM)
    return global_ddot


def GMRES(A, b, x, comm, rtol=1e-4, maxiter=50):
    extra = (0 if comm.Get_rank() < A.shape[1] % comm.Get_size() else 1)
    row = A.shape[0] + extra
    r = np.zeros(row)
    r[:b.size] = b
    matvec(A, x, r, comm, -1)
    r0 = ddot(r, r, comm) ** 0.5
    Q = np.copy(r)[:, None] / r0
    h = np.zeros((maxiter + 1, maxiter))

    Aq = np.zeros(row)
    q = r
    for n in range(maxiter):
        matvec(A, q, Aq, comm, -1)
        for j in range(n):
            h[j, n] = np.dot(Q[:, j], Aq)
            Aq -= h[j, n] * Q[:, j]
        h[n+1, n] = norm(Aq)
        q = Aq / h[n+1, n]
        Q = np.append(Q, q[:, None], axis=1)
        e1 = np.zeros(n+2)
        e1[0] = r0
        Q_h, R = np.linalg.qr(h[:n+2, :n+1], mode='reduced')
        Qr = np.matmul(Q_h.T, e1)
        y = np.linalg.solve(R,Qr)
        res = norm(h[:n+2, :n+1] @ y - e1)
        if res / norm(e1) < rtol:
            x = Q[:, :n+1] @ y
            return
    x = Q[:, :n+1] @ y


def matvec(A, x_recv, res, comm, alpha=1):
    col = A.shape[1]
    rank = comm.Get_rank()
    size = comm.Get_size()
    offset = (col // size) * rank + (rank if rank < col % size else col % size)
    dest = (rank + size - 1) % size
    source = (rank + 1) % size
    x_send = np.copy(x_recv)
    
    res[:] = 0
    for k in range(size):
        col_block = col // size + (1 if ((k + rank) % size < col % size) else 0)
        req_r = comm.Irecv(x_recv, source, 21)
        req_s = comm.Isend(x_send, dest, 21) 
        res[:A.shape[0]] += alpha * (A[:, offset : offset + col_block] @ x_send[:col_block])
        offset = (offset + col_block) % col
        req_r.wait()
        req_s.wait()
        np.copyto(x_send, x_recv)
        comm.Barrier()


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    args = sys.argv
    n = int(args[1])

    m = n // size + (1 if rank < n % size else 0)
    extra = (0 if rank < n % size else 1) # для выравнивания памяти
    start = rank * (n // size) + min(n % size, rank)
    A = np.fromfile("A.npy", count = m * n, offset = start * n * 8).reshape(m, -1)
    b = np.fromfile("b.npy", count = m, offset = start * 8)

    x_0 = np.random.rand(m + extra)
    res = np.random.rand(m)
    comm.Barrier()
    start_time = time.time()
    GMRES(A, b, x_0, comm, maxiter=100)
    comm.Barrier()
    end_time = time.time()
    if rank == 0:
        print(end_time - start_time)
    MPI.Finalize()

