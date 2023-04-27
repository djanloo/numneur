import numpy as np
from .networks import parents_and_children
from numpy import tanh, cosh
cdef mystic_function(v):
  """Directly from the article"""
  return 0.04*v*v + 5.0*v + 140.0

def neuronet(double [:] I, double [:,:] g0, double [:,:] Esyn,
                      double a=0.02, double b=0.2,double  c=-65.0, double d=6.0, 
                      double dt=0.001, injection_neuron=0):
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
                            + dt*( mystic_function(v[neuron_index, t])\
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


    


