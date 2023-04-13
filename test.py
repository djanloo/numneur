from numneur.neur import izhikevich_RK as izhiRK, izhikevich as izhi, syn_izhikevich_RK as sizi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.size"] = 9
rcParams["font.family"] = "serif"

fig = plt.figure(constrained_layout=True)
axs = fig.subplot_mosaic([['V'],["I"], ["g"]],
                          gridspec_kw={'height_ratios':[3, 1, 1]}, sharex=True)
N = 50_000
dt = 1e-2
T = np.arange(N)
t = np.arange(N)*dt


chattering = dict(c=-50.0, d=2)
bursting =   dict(c=-55.0, d=4)
accomodating = dict(a=0.02, b=1, c=-65, d=2)

I = np.zeros(N)
I[N//40:N//5*4] = 10
# I[N//5: N//5 + 5000] = -10
# I = 30*np.sin(t/50)
v, recovery, firing_times = izhiRK(I, dt=dt, **chattering)

axs["V"].plot(t, v, label="Neuron 1")
axs["I"].plot(t, I, label="Neuron 1 (injected)")

g = np.zeros(len(I))
tau = 10
t_retard = 0.0
g0 = 1.0
for ft in firing_times:
    t_shift = ft*dt + t_retard
    mask = (t >= t_shift)
    g[mask] += g0*np.e*(t[mask] - t_shift)/tau*np.exp( -(t[mask] - t_shift)/tau)
    axs["g"].axvline(ft*dt, lw=0.5, color="k", alpha=0.2)
axs["g"].plot(t, g)
Esyn = -80.0

v, u, firing_times = sizi(g, Esyn = Esyn, **bursting)

axs["V"].plot(t, v, label="Neuron 2")
axs["I"].plot(t, g*(v-Esyn), label="Neuron 2 (synaptic)")


axs["V"].legend()
axs["V"].set_ylabel("V")

axs["I"].legend()
axs["I"].set_ylabel("I")
axs["g"].set_ylabel("g")
plt.show()