"""Profiling examle using line_profiler.
"""
from os import chdir
from os.path import dirname, join
from line_profiler import LineProfiler
from numneur import neur, networks

import numpy as np
import matplotlib.pyplot as plt

from rich import print

# Sets the working directory as the one with the code inside
# Without this line line_profiler won't find anything
chdir(join(dirname(__file__), "numneur"))

lp = LineProfiler()

lp.add_function(neur.neuronet)
lp.add_function(networks.watts_strogatz)
lp.add_function(networks.parents_and_children)


wrap = lp(neur.neuronet)

dt = 1e-2
I = 10*np.ones(10_000)
M = 10
# g0 = np.random.uniform(0,0.5, size=(M,M))

adj = networks.watts_strogatz(M, 6, .1)
for i in range(M):
    adj[i,i] = 0
g0 = adj*np.random.uniform(0,.5, size=(M,M))

bursting = dict(c=-55.0, d=4)



v, f = wrap(I, g0, Esyn=0.0, dt=dt ,**bursting)

for vv in v:
    plt.plot(vv)

lp.print_stats()

