import numpy as np
from libc.stdlib cimport rand

cdef extern from "limits.h":
    int INT_MAX

cdef float randzerone():
  return rand()/ float(INT_MAX)

cdef int mod (int a, int b):
    cdef int r = a % b
    if r < 0:
        r += b
    return r

def watts_strogatz(int N, int K, double beta):
    """Watts-Strogatz small-world network generator."""
    cdef double [:,:] A = np.zeros((N,N))
    cdef int i, j, left, right, attempt, connection
    cdef bint connected = False

    for i in range(N):
        for j in range(1, K//2+1):
            left = mod(i-j, N)
            right = mod(i+j, N)
            A[i, left]  = 1.0
            A[i, right] = 1.0
            A[i, i] += 2.0

    for i in range(N):
        connected = False

        if randzerone() < beta:
            while not connected:
                connection = rand()%N

                if A[i, connection] == 0.0:

                    A[i, connection] = 1.0
                    A[connection,i] = 1.0

                    A[connection, connection] += 1

                    A[i, mod(i + K//2, N)] = 0
                    A[mod(i+K//2, N), i] = 0

                    A[mod(i + K//2, N), mod(i + K//2, N)] -= 1

                    connected = True
    
    return np.array(A)

def parents_and_children(double [:,:] G):
    cdef int N = len(G), i, j, n_parents, n_children

    # Takes trace the max number of parents and children
    # to return a memoryview (maxparents, maxchildren)
    cdef int maxparents = -1, maxchildren = -1
    cdef int [:,:] parents, children

    for i in range(N):
        n_parents, n_children = 0, 0
        for j in range(N):
            if G[i, j] != 0 and i != j:
                n_children += 1
            if G[j, i] != 0 and i != j:
                n_parents += 1
        maxchildren = n_children if n_children > maxchildren else maxchildren
        maxparents = n_parents if n_parents > maxparents else maxparents
    
    parents  = -np.ones((N, maxparents),    dtype="intc")
    children = -np.ones((N, maxchildren),   dtype="intc")
        
    for i in range(N):
        n_parents, n_children = 0, 0
        for j in range(N):

            if G[i, j] != 0 and i != j:
                children[i, n_children] = j
                n_children += 1

            if G[j, i] != 0 and i != j:
                parents[i, n_parents] = j
                n_parents += 1
    
    return parents, children
