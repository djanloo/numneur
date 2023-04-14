import numpy as np


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