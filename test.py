from numneur.neur import izhikevich_RK as izi
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()

I = np.zeros(50_000)
I[1000:30_000] = 10
I[0:500] = -1.0
noise = np.random.normal(0,0.01, size=len(I) )
I += np.cumsum(noise)



v, u = izi(I,  c = -50.0, d= 2,dt=1e-2)
plt.plot(v, label="V")
plt.plot(u, label="recovery")
plt.plot(I, label="injected current")

plt.legend(fontsize=10)
plt.show()