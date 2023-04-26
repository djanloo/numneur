from numneur.neur import morris_lecar_oscillator as ml
import numpy as np
import matplotlib.pyplot as plt

parameters = dict(
                C=20, 
                v_ca = 120, v_k = -84, v_l=-60, 
                phi=0.04, 
                g_ca=47.7, g_k=20, g_l= 0.3,
                v_1 = -1.2, v_2 = 18, v_3 = 2, v_4 = 30
                )
I = 200*np.ones(10_000)
v, n = ml(I, -100 , 0, 1e-2, **parameters)
plt.plot(v)
plt.plot(n)
plt.show()