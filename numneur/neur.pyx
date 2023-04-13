import numpy as np


cdef mystic_function(v):
  """Directly from the article"""
  return 0.04*v*v + 5.0*v + 140.0

def izhikevich(double [:] I, double a=0.02, double b=0.2,double  c=-65.0, double d=6.0, double dt=0.001):
  """Integration using explicit Euler"""
  cdef int N = len(I), i
  cdef double [:] v = np.zeros(N)
  cdef double [:] u = np.zeros(N)
  cdef list firing_times = []

  # Initial condition: Reset potential
  v[0] = c

  for i in range(N-1):

    # Fire and reset
    if v[i] >= 30.0:
      v[i] = c
      u[i] = u[i] + d
      firing_times.append(i)

    v[i+1] = v[i] + dt*( mystic_function(v[i]) - u[i] + I[i])
    u[i+1] = u[i] + a*dt*( b*v[i] - u[i])
  
  return np.array(v), np.array(u), firing_times


def izhikevich_RK(double [:] I, double a=0.02, double b=0.2,double  c=-65.0, double d=6.0, double dt=0.001):
  """Integration by a fourth order RUnge Kutta"""
  cdef int N = len(I), i
  cdef double [:] v = np.zeros(N)
  cdef double [:] u = np.zeros(N)
  cdef list firing_times = []

  cdef double k1, k2, k3, k4

  # Initial condition: Reset potential
  v[0] = c


  for i in range(N-1):

    # Fire and reset
    if v[i] >= 30.0:
      v[i] = c
      u[i] = u[i] + d
      firing_times.append(i)

    # Midtime estimates of u
    midtime_u = u[i] + a* 0.5*dt*(b*v[i] - u[i])
    fulltime_u = u[i] + a*dt*(b*v[i] - u[i])

    # Runge-Kutta
    k1 = dt*(mystic_function(v[i]) - u[i])
    k2 = dt*(mystic_function(v[i] + 0.5*k1) - midtime_u )
    k3 = dt*(mystic_function(v[i] + 0.5*k2) - midtime_u )
    k4 = dt*(mystic_function(v[i] +     k3) - fulltime_u)

    v[i+1] = v[i] + k1/3.0 + k2/6.0 + k3/6.0 + k4/3.0 +  dt*I[i]
    u[i+1] = u[i] + a*dt*( b*v[i+1] - u[i])
  
  return np.array(v), np.array(u), firing_times