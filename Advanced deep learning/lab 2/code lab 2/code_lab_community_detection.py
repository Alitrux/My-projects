

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans



############## Task 6
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    
    A = nx.adjacency_matrix(G)
    inv_degree_sequence = [1/G.degree(node) for node in G.nodes()]
    D_inv = diags(inv_degree_sequence)
    n = G.number_of_nodes()
    Lrw = eye(n) - D_inv @ A
    _, eigenvectors = eigs(Lrw, k=k, which='SR')
    eigenvectors = eigenvectors.real

    k_means = KMeans(n_clusters=k)
    k_means.fit(eigenvectors)

    clustering = dict()
    for i, node in enumerate(G.nodes()):
        clustering[node] = k_means.labels_[i]

    return clustering





############## Task 7

G = nx.read_edgelist("./CA-HepTh.txt", comments='#', delimiter='\t')
clustering = spectral_clustering(G, 50)
print(clustering, "\n\n")





############## Task 8
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    m = G.number_of_edges()
    clusters = set(clustering.values())
    modularity = 0

    for cluster in clusters:
        cluster_nodes = [node for node in G.nodes() if clustering[node] == cluster]
        cluster_edges = [(node1, node2) for node1 in cluster_nodes for node2 in cluster_nodes if G.has_edge(node1, node2)]
        cluster_degree = sum([G.degree(node) for node in cluster_nodes])
        cluster_degree /= 2
        cluster_modularity = len(cluster_edges)/m - (cluster_degree/(2*m))**2
        modularity += cluster_modularity

    return modularity



############## Task 9

# Compute modularity for spectral clustering result
modularity_spectral = modularity(G, clustering)
print("Modularity of spectral clustering:", modularity_spectral)

# Randomly partition nodes into 50 clusters
random_clustering = {node: randint(0, 49) for node in G.nodes()}

# Compute modularity for random clustering result
modularity_random = modularity(G, random_clustering)
print("Modularity of random clustering:", modularity_random)







