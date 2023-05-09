import numpy as np
import matplotlib.pyplot as plt
import pyNN.neuron as sim
from pyNN.random import NumpyRNG, RandomDistribution
from pyNN.utility import ProgressBar
from pyNN.utility.plotting import plot_spiketrains, plot_signals

simulation_time = 30

# Neurons
population = sim.Population(100,sim.EIF_cond_alpha_isfa_ista())
population.record(("v", "spikes"))

# Connections
synapse = sim.StaticSynapse(weight=0.01, 
                            delay=15)

random_conn = sim.FixedProbabilityConnector(0.5)
a2a_conn = sim.AllToAllConnector()

connections = dict(self_connections = sim.Projection(population, population, 
                                                    random_conn, synapse, 
                                                    receptor_type='excitatory'))

# Injection
pulse = sim.DCSource(amplitude=5.0, start=0.0, stop=50.0)
# noisy = sim.NoisyCurrentSource(mean=50.5, stdev=1.0, start=0.0, stop=450.0, dt=0.01)
pulse.inject_into(population.sample(1))

def callback():
    pb = ProgressBar()
    def increment(elapsed):
        elapsed += 1/simulation_time
        pb(elapsed)
        return elapsed
    return increment


sim.run(simulation_time, callbacks=[callback()])
print()

# Plot data
block = population.get_data()
print(f"{len(block.segments)} segments of block found")
print(block.segments[0].analogsignals)
print(block.segments[0].spiketrains)

fig, (axspike, axv) = plt.subplots(2,1, sharex=True)
for signal in block.segments[0].analogsignals:
    axv.plot(signal.times, signal.magnitude)
plot_spiketrains(axspike, block.segments[0].spiketrains)

plt.show()