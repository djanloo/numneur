import import_context
from numneur.neur import morris_lecar_oscillator as ml
import numpy as np
import matplotlib.pyplot as plt

hopf_bifurcation = dict(
                C=20, 
                v_ca = 120, v_k = -84, v_l=-60, 
                phi=0.04, 
                g_ca= 4.4, g_k = 8.0, g_l= 2.0,
                v_1 = -1.2, v_2 = 18, v_3 = 2.0, v_4 = 30
                )

snic_bifurcation = dict(
                C=20, 
                v_ca = 120, v_k = -84, v_l=-60, 
                phi=0.067, 
                g_ca= 4.0, g_k = 8.0, g_l= 2.0,
                v_1 = -1.2, v_2 = 18, v_3 = 12.0, v_4 = 17.4
                )

I = 89.5*np.ones(100_000)

v, n = ml(I, -52 , 0.02, 1e-2, **hopf_bifurcation)
plt.plot(v, n)
plt.figure(2)
plt.plot(v)

plt.show()