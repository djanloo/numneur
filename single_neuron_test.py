from numneur.neur import izhikevich as izhi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.size"] = 9
rcParams["font.family"] = "serif"

# Parameters from  Izhikevich's article
chattering = dict(c=-50.0, d=2)
bursting =   dict(c=-55.0, d=4)
accomodating = dict(a=0.02, b=1, c=-65, d=2)

N = 100_000
dt = 1e-2

T = np.arange(N)
t = np.arange(N)*dt


# Injected current
I = -20*np.ones(N)


# Computes the potential of the first neuron
v, u, firing_times = izhi(I, dt=dt, **chattering)

fig, (axt, psth) = plt.subplots(2,1)
axt.plot(t, v)
axt.plot(t, u)
print(np.diff(firing_times))
psth.hist(np.diff(firing_times))

plt.show()