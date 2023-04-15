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
    # Dummy comment 3
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
        # print("pre")
        # print(np.array(A))
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
                    # print(f"connected {i} with {connection}, removed connection ({i}, {mod(i+K//2, N)})")
            # print("post")
            # print(np.array(A))
    
    return np.array(A)