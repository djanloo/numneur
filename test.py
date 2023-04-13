from numneur.neur import izhikevich_RK as izhiRK, izhikevich as izhi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.size"] = 9
rcParams["font.family"] = "serif"

fig = plt.figure(constrained_layout=True)
axs = fig.subplot_mosaic([['V'],["I"], ["g"]],
                          gridspec_kw={'height_ratios':[3, 1, 1]}, sharex=True)
N = 50_000
dt = 2e-2
T = np.arange(N)
t = np.arange(N)*dt


chattering = dict(c=-50.0, d=2)
bursting =   dict(c=-55.0, d=4)
accomodating = dict(a=0.02, b=1, c=-65, d=2)

I = np.zeros(N)
I[N//50:N//5*3] = 10
I[N//5: N//5 + 5000] = -10
# I = 30*np.sin(t/50)
v, recovery, firing_times = izhiRK(I, dt=dt, **bursting)

axs["V"].plot(t, v, label="RK4")
axs["I"].plot(t, I, label="injected current")

g = np.zeros(len(I))
tau = 2
for ft in firing_times:
    mask = t > ft*dt
    g[mask] += 10*(t[mask] - ft*dt)/tau*np.exp( -(t[mask]-ft*dt)/tau)
axs["g"].plot(t, g)


v = izhi(I, dt=dt, **bursting)
axs["V"].plot(t, v, label="eEuler")
axs["V"].legend()

axs["V"].set_ylabel("Membrane potential")
axs["I"].set_ylabel("I")
axs["g"].set_ylabel("g")
fig.suptitle("Izhikevich model (chattering pattern)")
plt.show()