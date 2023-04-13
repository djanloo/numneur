"""Profiling examle using line_profiler.
"""
from os import chdir
from os.path import dirname, join
from line_profiler import LineProfiler
from numneur import neur
import numpy as np

from rich import print

# Sets the working directory as the one with the code inside
# Without this line line_profiler won't find anything
chdir(join(dirname(__file__), "numneur"))

lp = LineProfiler()

lp.add_function(neur.izhikevich_RK)
wrap = lp(neur.izhikevich_RK)

I = 10*np.ones(50_000)
wrap(I)

lp.print_stats()

lp = LineProfiler()

lp.add_function(neur.izhikevich)
wrap = lp(neur.izhikevich)

I = 10*np.ones(500_000)
wrap(I)

lp.print_stats()