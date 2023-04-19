from numneur.networks import barabasi_albert as ba
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from numneur.utils import split_nodes

adj = ba(100, 5, 1.0)
print(adj)
diag = adj.diagonal().copy()

# for i in range(len(adj)):
#     adj[i,i] = 0

graph = nx.Graph(split_nodes(adj,1))
nx.draw_spring(graph, node_size=30, width=0.3, node_color=np.concatenate((np.log(diag), -1*np.ones(len(diag)))))

plt.figure(2)
plt.hist(diag)
plt.yscale("log")

plt.figure(3)
plt.matshow(adj)
plt.show()