import numpy as np
from numneur.neur import neuronet
from numneur.networks import directed_small_world as dsw
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
from matplotlib import rcParams

np.random.seed(42)
rcParams["font.size"] = 9
rcParams["font.family"] = "serif"
fig = plt.figure(constrained_layout=True)
axs = fig.subplot_mosaic([['graph',"info"],["V","V"], ["A", "A"]],
                          gridspec_kw={'height_ratios':[1, .8, .5]}, 
                          sharex=False)
# Neuron parameters
bursting = dict(c=-55.0, d=4)
chattering = dict(c=-50.0, d=2)
accomodating = dict(a=0.02, b=1, c=-65, d=2)
tonic = dict(a=0.02, b=0.2, c=-65, d=6)

# Simulations parameters
T = 10_000
dt = 2e-2
t = np.linspace(0, T*dt, T)
print("max time:", T*dt)

# Current stimulus for neuron 0
I = 100*np.ones(T)
I[0: 100] = 0.0

M = 500 #Number of neurons

# Synaptic stuff
conductance_weights = np.random.uniform(0.01, 1.0, size=(M,M))
synaptic_potentials = np.random.uniform(0,1, size=(M,M))
excit_inhib = 0.2 # proportions of inhibitory/excitatory
excitatory = synaptic_potentials > excit_inhib
inhibitory = synaptic_potentials <= excit_inhib
synaptic_potentials[excitatory] = 0
synaptic_potentials[inhibitory] = -90

# Small world network generation
s_w_prob = 0.5 # small-world probability
forward_neighbors = 6
adj = dsw(M, forward_neighbors, s_w_prob)

# Set diagonal to zero in case they are not
for i in range(M):
    adj[i,i] = 0

# Counts synapses
synapses = adj > 0
excitatory = excitatory & synapses
inhibitory = inhibitory & synapses

# Sets the generated conductances to synapses
g0 = adj*conductance_weights
print("generation done")

# Useless part: necessary to plot a (more or less) nice graph
g0_inv = g0.copy()
mask = (g0_inv == 0.0)
g0_inv[mask] += 1
g0_inv = 1.0/g0_inv
g0_inv[mask] = 0.0

# Graph plot
graph = nx.Graph(g0_inv)
pos = nx.spectral_layout(graph)
pos_array = np.zeros((M,2))
for n in pos.keys():
    pos_array[n] = pos[n]

print("embedding done")

# Simulate
v, f = neuronet(I, g0, Esyn=synaptic_potentials, dt=dt ,**tonic)
print("simulation done")


# spiketimes, indexes = np.flip(f).T
# axt.plot(spiketimes*dt, indexes, ls="", marker="|", color="k")
# tt, nn = np.meshgrid(t, np.arange(M))


## Graph plot settings
nx.draw_networkx_edges(graph, pos, width=0.5, ax=axs["graph"])
scat = axs["graph"].scatter(*pos_array.T, c=np.zeros(len(g0)), s=20, vmin=-90, vmax=-30, cmap="plasma")
axs["graph"].set_xlim(-1,1)
axs["graph"].set_ylim(-1,1)
axs["graph"].set_aspect("equal")

## V plot
timeline = axs["V"].axvline(0, color="r")
axs["V"].matshow(v)
axs["V"].set_ylim(-1, M+1)
axs["V"].set_aspect("auto")
axs["V"].set_ylabel("$V_i$")

## Activity
spikeindex, n_index = np.flip(f).T
axs["A"].hist(spikeindex, histtype="step", color="k", bins=T//50)
timeline2 = axs["A"].axvline(0, color="r")
axs["A"].set_ylabel("PSTH")

## infos
axs["info"].annotate(f"neurons = {M}", (0,0.0))
axs["info"].annotate(f"forward neighbors = {forward_neighbors}", (0,-.5))
axs["info"].annotate(f"total synapses = {np.sum(synapses)} ( {np.sum(synapses)/(M**2 - M)*100:.1f}% connettivity)", (0, -1))
axs["info"].annotate(f"inhibitory synapses = {np.sum(inhibitory)/np.sum(synapses)*100:.1f} %", (0,-1.5))
axs["info"].annotate(f"small-world probability = {s_w_prob*100:.1f} %", (0,-2))
axs["info"].set_ylim(-4, 1)
axs["info"].set_xlim(0,3)
axs["info"].axis("off")

# Animation parameters
startframe = 1000
frames = 1200
speed = 5

# Time indicators
axs["V"].axvline(startframe, color="k")
axs["V"].axvline(startframe + speed*frames, color="k")

def update(i):
    tt = speed*i + startframe
    scat.set_array(v[:, tt])
    timeline.set_data([tt,tt], [0,1])
    timeline2.set_data([tt,tt], [0,1])

    print(f"{i/frames*100:.1f}", end=" ", flush=True)
    return scat, timeline, timeline2

anim = FuncAnimation(fig, update, frames=frames, interval=16, blit=True)
# anim.save("dummy.mp4")
plt.show()