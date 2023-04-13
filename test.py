from numneur.neur import izhikevich_RK as izhiRK, izhikevich as izhi, syn_izhikevich_RK as sizi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.size"] = 9
rcParams["font.family"] = "serif"

fig = plt.figure(constrained_layout=True)
axs = fig.subplot_mosaic([['V'],["u"], ["I"], ["g"]],
                          gridspec_kw={'height_ratios':[3, 2, 1, 1]}, sharex=True)
N = 50_000
dt = 1e-2
T = np.arange(N)
t = np.arange(N)*dt

# Parameters from  Izhikevich's article
chattering = dict(c=-50.0, d=2)
bursting =   dict(c=-55.0, d=4)
accomodating = dict(a=0.02, b=1, c=-65, d=2)

# Injected current
I = np.zeros(N)
I[N//40:N//5*4] = 10

# Computes the potential of the first neuron
v1, recovery1, firing_times = izhiRK(I, dt=dt, **chattering)

# Synapse stuff
tau = 10
t_retard = 0.0
g0 = 1.0
Esyn = 0.0

g = np.zeros(len(I))
for ft in firing_times:
    t_shift = ft*dt + t_retard
    mask = (t >= t_shift)
    g[mask] += g0*np.e*(t[mask] - t_shift)/tau*np.exp( -(t[mask] - t_shift)/tau)
    axs["g"].axvline(ft*dt, lw=0.5, color="k", alpha=0.2)

axs["g"].plot(t, g)

# Computes the potential of the second neuron
v2, recovery2, firing_times = sizi(g, Esyn = Esyn, **bursting)


# plots
axs["V"].plot(t, v1, label="Neuron 1")
axs["I"].plot(t, I, label="Neuron 1 (injected)")

axs["V"].plot(t, v2, label="Neuron 2")
axs["I"].plot(t, g*(v2-Esyn), label="Neuron 2 (synaptic)")

axs["u"].plot(t, recovery1, label="Neuron1")
axs["u"].plot(t, recovery2, label="Neuron2")

axs["V"].legend()
axs["V"].set_ylabel("V")

axs["I"].legend()
axs["I"].set_ylabel("I")
axs["g"].set_ylabel("g")
axs["u"].set_ylabel("u")

plt.show()