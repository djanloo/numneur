import numpy as np
from numneur.neur import double_neuron
import matplotlib.pyplot as plt

T = 50_000
dt = 1e-2

I = 5*np.ones(T)
I[0: 1000] = 0.0


g0 = np.array([[0.0, 1.0, 0.0, 0.3], 
               [0.0, 0.0, 1.0, 0.3],
               [0.0, 0.0, 0.0, 0.3],
               [0.6, 0.0, 0.0, 0.0]])

M = 20
g0 = np.random.uniform(0,0.5, size=(M,M))

bursting = dict(c=-55.0, d=4)
chattering = dict(c=-50.0, d=2)
accomodating = dict(a=0.02, b=1, c=-65, d=2)


v, f = double_neuron(I, g0, Esyn=0.0, dt=dt ,**bursting)
# print(f)
for n_index in range(len(v)):
    plt.plot(v[n_index,:])

plt.figure(2)

for firing in f:
    plt.scatter([firing[1]], [firing[0]], color="k", marker="|")


fig, ax = plt.subplots()
from matplotlib.animation import FuncAnimation

scat = ax.scatter(*np.random.uniform(0,1, size=(2, M)), c=np.zeros(M), s=300, vmin=-90, vmax=30)
ax.set_xlim(0,1)
ax.set_ylim(0,1)
def update(i):
    scat.set_array(v[:, 100*i])
    print(i)
    # scat.set_offsets(np.random.uniform(0,1, size=(2, M)))

anim = FuncAnimation(fig, update, frames=1000, interval=10)

plt.show()