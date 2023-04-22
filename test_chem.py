import numpy as np
import matplotlib.pyplot as plt
from numneur.chem import gillespie

reaction = "[H2O]       --> [H] + [O] + [O]     |0.5|#\
            [O] + [Am]  --> [AmO]          |0.1|#\
            [H]         --> [X]            |0.01|#\
            [AmO]       --> [O] + [Am]     |0.3|"

equilibrium = " [AgCl]      --> [Ag] + [Cl] |0.5|#\
                [Ag] + [Cl] --> [AgCl]      |0.01|"

initial_state = {"AgCl": 0, "Cl": 10000, "Ag": 1002}
t, evol = gillespie(equilibrium, initial_state, N=10_000)

for species in evol.keys():
    plt.plot(t, evol[species], label=species)
plt.legend()
plt.show()
