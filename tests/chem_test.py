import import_context
import numpy as np
import matplotlib.pyplot as plt
from numneur.chem import gillespie

reaction = "[H2O] --> [H] + [O] + [O]     |0.5|"

equilibrium = " [AgCl]      --> [Ag] + [Cl] |0.5|#\
                [Ag] + [Cl] --> [AgCl]      |0.01|"

initial_state = {"H2O": 1000, "O":0.0, "H": 0.0}
t, evol = gillespie(reaction, initial_state, N=2_000)

for species in evol.keys():
    plt.plot(t, evol[species], label=species)
plt.legend()
plt.show()
