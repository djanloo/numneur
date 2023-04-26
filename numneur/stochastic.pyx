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


def thresholded_OU(double [:] I, thr=1.0, reset=0.0, dt=1e-2, sigma=1.0, R=1.0, tau=1.0):
    """Integrates the Ornstein Uhlenbeck process with a threshold"""

    cdef int N = len(I), i
    cdef double [:] v = np.zeros(N)
    cdef double [:] noise = np.random.normal(0, sigma*np.sqrt(dt), size=N)
    cdef list firing_times = []

    v[0] = reset
    for i in range(N-1):
        v[i+1] = v[i] + dt/tau*( - v[i] + R*I[i] ) + noise[i]
        if v[i + 1] >= thr:
            v[i+1] = reset
            firing_times.append(i)

    return np.array(v), firing_times


def poisson_process(nu, N, double dt):
    cdef int i, interspike_counter
    cdef int [:] P = np.zeros(N, dtype="intc")
    cdef list interspike = []

    interspike_counter = 0
    for i in range(N):
        interspike_counter += 1 
        if randzerone() < nu*dt:
            P[i] = 1
            interspike.append(interspike_counter*dt)
            interspike_counter = 0
    return np.array(P), interspike
