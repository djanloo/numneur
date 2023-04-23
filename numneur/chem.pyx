"""Module for the analysis of molecular species.
"""

import cython
import numpy as np
from libc.math cimport log
from libc.stdlib cimport rand
import re

cdef extern from "limits.h":
    int INT_MAX

cdef float randzerone():
  return rand()/ float(INT_MAX)

def chemreact_from_string(string_in):
    """From a string to a chemical reaction.
    A reaction has format:
    
    [A] + [B] --> [C]       |k1|   #
    [C] + [D] --> [A] + [B] |k2|

    Reverse reactions must be specified in different lines. Also,
    stechiometric coefficients are not supported yet.

    A reaction is simply a dictionary of reagents, products and a kinetic constant.
    """
    lines = string_in.split("#")
    species = set()
    reactions = []

    for line in lines:
        divide = re.match(r"(.+?)-->(.+?)\|(.+?)\|", line)
        if divide is not None:
            reagents = re.findall(r"\[(.+?)\]", divide.group(1))
            products = re.findall(r"\[(.+?)\]", divide.group(2))
            const = float(divide.group(3))

            species = species.union(set(reagents))
            species = species.union(set(products))

            reactions.append(dict(reagents=reagents, products=products, const=const))
        else:
            print(f"The following reaction could not be interpreted: \n{line}")
    
    symbol_map = {molecule:i for i, molecule in enumerate(list(species))}

    for i,r in enumerate(reactions):
        print(f"Reaction {i}: {'+'.join(r['reagents']):10} --> {'+'.join(r['products']):10} with k = {r['const']:.2f}")

    # Converts symbols to integers
    for r in reactions:
        r["reagents"] = [symbol_map[spec] for spec in r["reagents"]]
        r["products"] = [symbol_map[spec] for spec in r["products"]]
        
    return reactions, symbol_map


def gillespie(str reactions_str, dict initial_state , N=1000, V=10.0):
    """Gillespie SSA.
    
    Takes a string of reactions as in chemreact_from_string and a dictionary 
    of initial state for each molecular species.

    V is the volume, N the number of reactions to simulate (not the time).
    Probably V is useless.
    """
    cdef int i, species, j
    cdef double [:] reaction_t = np.zeros(N)

    # Gets the chemical reaction fro the string
    reactions, symbol_map = chemreact_from_string(reactions_str)
    n_species = len(symbol_map)
    n_reactions = len(reactions)

    cdef double [:,:] population = np.zeros((N, n_species)) 

    # Set initial conditions
    for species_symb in initial_state.keys():
        population[0, symbol_map[species_symb]] = initial_state[species_symb]

    # Reaction probability coefficients
    cdef double [:] a = np.zeros(n_species)

    for i in range(1, N):

        # Computes the "a" coefficients
        for r in range(n_reactions):
            concentrations_product = 1.0
            for reagent in reactions[r]["reagents"]:
                concentrations_product *= population[i-1, reagent]

            a[r] = reactions[r]["const"]*concentrations_product

        a0 = np.sum(a)

        if a0 == 0.0:
            print(f"Reagent exhaurited at step {i} over {N}")
            break

        # Generates the reaction time
        tau = -log(randzerone())/a0
        reaction_t[i] = tau+reaction_t[i-1]

        # Copies the old state
        for species in range(n_species):
            population[i, species] = population[i-1, species]

        # Selects the reaction to use
        u = a0*randzerone()
        cumulative_coeff = 0.0

        for r in range(n_reactions):
            cumulative_coeff += a[r]
            if u < cumulative_coeff:
                selected_reaction = reactions[r]
                break

        for reagent in selected_reaction["reagents"]:
            population[i, reagent] -= 1
        
        for product in selected_reaction["products"]:
            population[i, product] += 1

    ## Conversion to dictionary
    evolution = dict()
    for symbol in symbol_map.keys():
        # If the reagents were finished, the i index is stopped at the last reaction
        evolution[symbol] = np.array(population[:i, symbol_map[symbol]])

    return np.array(reaction_t[:i]), evolution





