import numpy as np
from numneur.neur import neuronet
from numneur.networks import watts_strogatz as ws
import matplotlib.pyplot as plt
import networkx as nx
np.random.seed(42)
T = 10_000
dt = 5e-2

I = 5*np.ones(T)
I[0: 1000] = 0.0

M = 20

adj = ws(M, 4, 0.5)
print("generation done")

for i in range(M):
    adj[i,i] = 0

graph = nx.Graph(adj)
pos = nx.spectral_layout(graph)
pos_array = np.zeros((M,2))
for n in pos.keys():
    pos_array[n] = pos[n]

g0 = adj*np.random.uniform(0,0.5, size=(M,M))

bursting = dict(c=-55.0, d=4)
chattering = dict(c=-50.0, d=2)
accomodating = dict(a=0.02, b=1, c=-65, d=2)

v, f = neuronet(I, g0, Esyn=0.0, dt=dt ,**bursting)
print("simulation done")

from matplotlib.animation import FuncAnimation
fig, (axt, axanim) = plt.subplots(2, 1)

for vv in v:
    axt.plot(vv)

nx.draw_networkx_edges(graph, pos, width=0.5)

scat = axanim.scatter(*pos_array.T, c=np.zeros(len(g0)), s=100, vmin=-90, vmax=-30)
timeline = axt.axvline(0, color="r")

# axanim.set_xlim(min(pos_arr),1)
# axanim.set_ylim(0,1)

def update(i):
    t = 2*i + 3300
    scat.set_array(v[:, t])
    timeline.set_data([t,t], [0,1])
    print(i)
    # scat.set_offsets(np.random.uniform(0,1, size=(2, M)))
    return scat, timeline

anim = FuncAnimation(fig, update, frames=500, interval=16)
anim.save("anim2.mp4")
# plt.show()