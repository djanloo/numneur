"""Module for simulation of neurons and networks of neurons.

Also, other dynamical models must be placed here (Morris-Lecar, Kuramoto).
"""
import numpy as np
from .networks import parents_and_children
from numpy import tanh, cosh

cdef Izk_pominomial(v):
  """Directly from the article"""
  return 0.04*v*v + 5.0*v + 140.0

def neuronet(double [:] I, double [:,:] g0, double [:,:] Esyn,
                      double a=0.02, double b=0.2,double  c=-65.0, double d=6.0, 
                      double dt=0.001, injection_neuron=0):
  """A network of IF neurons with exponential conductances.
  

  """
  cdef int T = len(I), i, j, t, neuron_index
  cdef int M = len(g0)

  cdef double [:,:] v = np.zeros((M, T))
  cdef double [:,:] u = np.zeros((M, T))

  # g[i, j] e' la conduttanza generata dal neurone i sul neurone j
  # nella evoluzione di v_n si usano quindi le g[i, n]
  # quando il neurone n scarica si incrementano le g[n, k]
  cdef double [:,:] g = np.zeros((M, M))

  cdef bint fired = False
  cdef list firing = []

  cdef int[:,:] parents, children 
  cdef int maxparents, maxchildren

  parents, children = parents_and_children(g0)
  
  maxparents  = len(parents[0])
  maxchildren = len(children[0])

  # Initialization at reset potential
  for neuron_index in range(M):
    v[neuron_index, 0] = c

  for t in range(T-1):
    if  100*t % T == 0:
      print("|", end="", flush=True)

    for neuron_index in range(M):
      fired = False
    
      v[neuron_index, t+1] =  v[neuron_index, t] \
                            + dt*( Izk_pominomial(v[neuron_index, t])\
                            - u[neuron_index ,t])

      # Synaptic currents
      for j in range(maxparents):
        if parents[neuron_index, j] == -1:
          break
        v[neuron_index, t+1] -= dt*g[parents[neuron_index, j], neuron_index]*(v[neuron_index, t] - Esyn[parents[neuron_index, j], neuron_index])

      # External stimulus
      if neuron_index == injection_neuron:
        v[neuron_index, t+1] += dt*I[t] 

      # Recovery variable
      u[neuron_index, t+1] =  u[neuron_index, t] \
                            + a*dt*( b*v[neuron_index, t] - u[neuron_index, t])

      ## FIRING
      if v[neuron_index, t+1] >= 30.0:
        v[neuron_index, t+1] = c 
        u[neuron_index, t+1] += d 
        firing.append([neuron_index, t])

        # Increment synaptic conductances of connected neurons
        for j in range(maxchildren):
          if children[neuron_index, j] == -1:
            break
          g[neuron_index, children[neuron_index, j]] += g0[neuron_index, children[neuron_index, j]]

    # Decay of all synapses
    for neuron_index in range(M):
      for j in range(maxchildren):
        if children[neuron_index, j] == -1:
          break
        g[neuron_index, children[neuron_index, j]] -= dt*g[neuron_index, children[neuron_index, j]]
      
  return np.array(v), firing

def morris_lecar_oscillator(I,v0, n0, dt, 
                            C = 0.1, v_1 = 0.1, v_2 = 0.1, v_3 = 0.1, v_4 = 0.1, phi = 0.1,
                            g_l = 0.1, g_ca = 0.1, g_k = 0.1,
                            v_l = 0.1, v_ca = 0.1, v_k = 0.1
                            ):

  cdef int i,j, N = len(I)

  cdef double [:] v = np.zeros(N)
  cdef double [:] n = np.zeros(N)

  v[0] = v0
  n[0] = n0

  for i in range(N-1):
    m_sat = 0.5*( 1 + tanh( (v[i] - v_1)/v_2    ))
    n_sat = 0.5*( 1 + tanh( (v[i] - v_3)/v_4    ))
    tau = 1.0 / ( cosh( (v[i] - v_3)/(2*v_4)))

    n[i+1] = n[i] + dt*phi/tau*(n_sat - n[i])
    v[i+1] = v[i] + dt/C*( I[i] - g_l*(v[i] - v_l) - g_ca*m_sat*(v[i] - v_ca) - g_k*n[i]*(v[i] - v_k))

  return np.array(v), np.array(n)

def izhikevich(I, dt=1e-3, **neuron_kwargs):
  """The single Ixhikevich neuron."""

  # If a parameter is not specified takes the tonic neuron as a reference
  tonic = dict(a=0.02, b=0.2, c=-65, d=6)
  neuron_kwargs = tonic | neuron_kwargs

  cdef int i, N = len(I)

  # Assigns the parameters
  cdef double a, b, c, d
  a,b,c,d = map(neuron_kwargs.get, ["a", "b", "c", "d"])

  cdef double [:] v = np.zeros(N), u = np.zeros(N)
  cdef list firing_times = []

  v[0] = c

  for i in range(N-1):

    # Updates
    v[i+1] = v[i] + dt*(0.04*v[i]*v[i] + 5.0*v[i] + 140.0 - u[i] + I[i])
    u[i+1] = u[i] + a*dt*( b*v[i] - u[i])

    # Firing
    if v[i+1] >= 30.0:
      v[i+1] = c 
      u[i+1] += d 
      firing_times.append(i*dt)
  
  return np.array(v), np.array(u), firing_times




    


