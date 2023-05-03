from numneur.neur import izhikevich as izhi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from rich.progress import track
import seaborn as sns

rcParams["font.size"] = 9
rcParams["font.family"] = "serif"

# Parameters from  Izhikevich's article
chattering = dict(c=-50.0, d=2)
bursting =   dict(c=-55.0, d=4)
accomodating = dict(a=0.02, b=1, c=-65, d=2)
tonic = dict(a=0.02, b=0.2, c=-65, d=6)

T = 500 # ms
M = 5

fig, (axt, ax_supp) = plt.subplots(2,1)
colors =sns.color_palette("flare", M)

for j, dt in track(enumerate(np.logspace(-2.5, -2, M))):
    N = int(T/dt)

    # Injected current
    I = 10*np.ones(N) + np.random.normal(0, 3/np.sqrt(dt),size=N)
    I[:N//10] = 0


    # Computes the potential of the first neuron
    v, u, firing_times = izhi(I, dt=dt, **chattering)

    t = np.arange(N)*dt
    axt.plot(t, v, color=colors[j], label=f"{np.log10(dt):.1f}")
    # axt.plot(t, u, color=colors[j])
    isi = np.diff(firing_times)
    isi = isi[-len(isi)//2:]
    print(np.mean(isi), np.std(isi))
    ax_supp.scatter([dt], [np.mean(isi)], color=colors[j], marker=".", s=10)
ax_supp.set_xscale("log")
axt.legend()
plt.show()