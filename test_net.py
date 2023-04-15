import numpy as np
from numneur.neur import neuronet
from numneur.networks import watts_strogatz as ws, parents_and_childs
import matplotlib.pyplot as plt
import networkx as nx
np.random.seed(42)
T = 10_000
dt = 5e-2

I = 20*np.ones(T)
I[0: 1000] = 0.0

M = 100

adj = ws(M, 6, .1)

for i in range(M):
    adj[i,i] = 0
g0 = adj*np.random.uniform(0,.5, size=(M,M))
print("generation done")

graph = nx.Graph(adj)
pos = nx.spectral_layout(graph)
pos_array = np.zeros((M,2))
for n in pos.keys():
    pos_array[n] = pos[n]


bursting = dict(c=-55.0, d=4)
chattering = dict(c=-50.0, d=2)
accomodating = dict(a=0.02, b=1, c=-65, d=2)
tonic = dict(a=0.02, b=0.2, c=-65, d=6)

v, f = neuronet(I, g0, Esyn=0.0, dt=dt ,**tonic)
print("simulation done")

from matplotlib.animation import FuncAnimation
fig, (axt, axanim) = plt.subplots(2, 1)

for vv in v:
    axt.plot(vv)

axt.plot(*np.flip(f).T, ls="", marker="|", color="k")

nx.draw_networkx_edges(graph, pos, width=0.5)

scat = axanim.scatter(*pos_array.T, c=np.zeros(len(g0)), s=50, vmin=-90, vmax=-30, cmap="plasma")
timeline = axt.axvline(0, color="r")


def update(i):
    t = 2*i + 2600
    scat.set_array(v[:, t])
    timeline.set_data([t,t], [0,1])
    # print(i)
    return scat, timeline

anim = FuncAnimation(fig, update, frames=600, interval=16, blit=True)
# anim.save("anim5.mp4")
plt.show()