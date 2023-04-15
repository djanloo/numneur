from numneur.networks import watts_strogatz as ws
import matplotlib.pyplot as plt
import networkx as nx

adj = ws(10, 4, 0.5)

diag = adj.diagonal().copy()

for i in range(len(adj)):
    adj[i,i] = 0

graph = nx.Graph(adj)
nx.draw_spectral(graph, node_size=30, width=0.2, node_color=diag, arrowsize=3)

plt.figure(2)
plt.hist(diag)
plt.yscale("log")
plt.show()