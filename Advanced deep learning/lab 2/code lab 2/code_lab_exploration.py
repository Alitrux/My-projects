import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1

G = nx.read_edgelist("./CA-HepTh.txt", comments='#', delimiter='\t')
print("number of nodes :", G.number_of_nodes())
print("number of edges :", G.number_of_edges())



############## Task 2

number_of_connected_components = nx.number_connected_components(G)
print("number of connected components : ", number_of_connected_components)

gcc_nodes = max(nx.connected_components(G), key=len)
gcc = G.subgraph(gcc_nodes)
print("number of nodes in GCC : ", gcc.number_of_nodes())
print("number of edges in GCC : ", gcc.number_of_edges())


############## Task 3
# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]

print("Average degree : ", np.mean(degree_sequence))
print("Maximum degree : ", max(degree_sequence))
print("Minimum degree : ", min(degree_sequence))




############## Task 4

plt.plot(nx.degree_histogram(G), label="Degree distribution")
plt.ylabel("Frequency")
plt.xlabel("Degree")
plt.legend()
plt.show()


plt.loglog(nx.degree_histogram(G), label="Degree distribution log-log scale")
plt.ylabel("log Frequency")
plt.xlabel("log Degree")
plt.legend()
plt.show()


############## Task 5

clustering_coefficient = nx.transitivity(G)
print("clustering coefficient :", clustering_coefficient)