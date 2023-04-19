import numpy as np

def split_nodes(G, lenghts):
    N = len(G)
    G_splitted = np.zeros((2*N, 2*N))
    G_splitted[N:, :N] = G
    G_splitted[:N, N:] = np.eye(N)*lenghts
    return G_splitted
