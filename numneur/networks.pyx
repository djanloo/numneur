"""Module for the construction of networks.

The most efficient way tested is the parent-children structure, 
that takes advantages from sparsely connected networks.
"""
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

def parents_and_children(double [:,:] G):
    """Utility to transform an adjacency matrix to a list of lists
    sparse structure.

    Memoryviews are currently used, but instances of python lists could be more efficient.
    """
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

                if connection != i and A[i, connection] == 0.0:

                    A[i, connection] = 1.0
                    A[connection,i] = 1.0

                    A[connection, connection] += 1

                    A[i, mod(i + K//2, N)] = 0
                    A[mod(i+K//2, N), i] = 0

                    A[mod(i + K//2, N), mod(i + K//2, N)] -= 1

                    connected = True
    
    return np.array(A)

def directed_small_world(int N, int K, double beta):
    """A test version of directed small-world netowork."""
    cdef double [:,:] A = np.zeros((N,N))
    cdef int i, j, right, connection
    cdef bint connected = False

    for i in range(N):
        for j in range(1, K):
            right = mod(i+j, N)
            A[i, right] = 1.0

    for i in range(N):
        connected = False

        if randzerone() < beta:
            while not connected:
                connection = rand()%N

                if connection!= i and A[i, connection] == 0.0:

                    # connects to new point
                    A[i, connection] = 1.0

                    # deletes rightermost link
                    A[i, mod(i + K-1, N)] = 0

                    connected = True
    
    return np.array(A)

def barabasi_albert(int N, int m0, double p0, double p_periphery):
    """The aggregative scale-invariant network of Barabasi-Albert"""
    cdef int [:,:] G = np.zeros((N,N), dtype="intc")
    cdef int Nlinks = 0
    cdef int i, j, t
    cdef list sectors = []
    cdef double cumulant_sector = 0.0
    # Initializes a m0-nodes random network 
    for t in range(m0):
        for j in range(t):
            if t != j:
                if randzerone() < p0:
                    # Connects i to j and j to i
                    G[t,j] = 1
                    G[j, t] = 1

                    # Increments the degree of node i and node j
                    G[t,t] += 1
                    G[j,j] += 1

                    # Increments number of links
                    Nlinks += 2

    for t in range(m0, N): # at each t a new node is generated

        # Sectors of the line to samples, e.g.:
        # if degrees are [1,1,2,1]
        # then sectors are [1/5, 2/5, 4/5, 1]
        # Sampling a random point on the (0,1) interval assigns the link 
        # with probability equal to the length of the segment

        sectors = []
        cumulant_sector = 0.0

        # print(t,Nlinks,  sectors)
        # print(np.array(G))

        for j in range(m0): # The linkage process is repeated m0 times

            for i in range(t):
                    cumulant_sector += G[i,i]/ (<int>Nlinks)
                    sectors.append(cumulant_sector)
            u = randzerone()
            # print(f"Extracted {u:.2f}")
            for i in range(t):
                if sectors[i] > u:
                    # print(f"selected sector {i}")
                    G[t, i] = 1
                    # G[i, t] = 1
                    
                    G[i, i] += 1
                    G[t, t] += 1

                    Nlinks += 2
                    break
        for i in range(N):
            for j in range(i):
                u = randzerone()
                if u < 1e-2:
                    G[i,j] = 1
                    G[j,j] += 1

    return np.array(G)

