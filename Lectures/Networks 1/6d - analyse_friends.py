"""
This script sets up an network of the people Donald Trump follows on Twitter
and performs standard network analysis on it. We consider various measures and 
attempt to detect communities.

The data are contained in "6b - names_ids.txt" and "6c - friends_of_friends_ids.txt".

The package we use is networkx. See the documentation here:
https://networkx.github.io/documentation/stable/reference/index.html
"""

###################################################################################################
# Preliminaries                                                                                   #
###################################################################################################

# Packages:
import matplotlib.pyplot as plt                   # For plotting.
import networkx as nx                             # For working with networks.
import numpy as np                                # A very popular library for scientific computations.
import pandas as pd                               # For dataframes.
from sklearn.cluster import KMeans                # k-means.

###################################################################################################
# Step 1: Loading in the data                                                                     #
###################################################################################################

# Reading in the names and ids:
with open('6b - names_ids.txt', 'rb') as f:
    names_ids_raw = f.readlines()
    
# Changing from bytes to string (for python 3):
names_ids_raw = [x.decode('utf-8') for x in names_ids_raw]

# Cleaning up:
names_ids = pd.DataFrame([x.replace('\n', '').split(',') for x in names_ids_raw])
names_ids.columns = ['names', 'ids']

# Reading in the ids of friends of friends:
with open('6c - friends_of_friends_ids.txt', 'rb') as f:
    friends_of_friends_ids_raw = f.readlines()
   
# Changing from bytes to string (for python 3):
friends_of_friends_ids_raw = [x.decode('utf-8') for x in friends_of_friends_ids_raw]
  
# Cleaning up:
friends_of_friends_ids = [x.replace('\n', '').split(',') for x in friends_of_friends_ids_raw]

###################################################################################################
# Step 2: Creating the network                                                                    #
###################################################################################################

# Create a directed graph:
G = nx.DiGraph()

# Add edges from Trump to everyone else:
for i in range(len(friends_of_friends_ids)):
    G.add_edge(0, i+1)

# Adding edges from the rest of the friends:
for i in range(len(friends_of_friends_ids)):
    for j in range(len(friends_of_friends_ids[i])):
        ind_of_friend = names_ids.index[names_ids['ids'] == friends_of_friends_ids[i][j]][0]
        G.add_edge(i+1, ind_of_friend)
    
# Setting up plot object and network layout:
fig, ax = plt.subplots(figsize = (10, 8))
#lay = nx.fruchterman_reingold_layout(G)
lay = nx.kamada_kawai_layout(G)

# Plotting:
nx.draw_networkx(G, lay, node_size = 150, width = 0.25, alpha = 0.75, edge_color = 'grey', node_color = 'purple', with_labels = False, ax = ax)

# We set the edge with to be quite thin to make the plot more readable. alpha sets the transparency of edges and nodes. See here for various plotting options:
# https://networkx.github.io/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html#networkx.drawing.nx_pylab.draw_networkx

# Drawing labels manually:
labels = names_ids['names']
lab_lay = {key:(val + 0.05) for key,val in lay.items()}
nx.draw_networkx_labels(G, lab_lay, labels, font_size = 10, font_color = 'blue')

# Showing the plot:
plt.show()

# Notce that "Greta" is unfortunately not Greta Thunberg.

# At first glance, do you notice any of the empirical regularities discussed in the slides here?

###################################################################################################
# Step 3: Analysing the network                                                                   #
###################################################################################################

# Undirected version of the graph:
G_u = G.to_undirected()

# We'll use the undirected version for certain measures.

# Number of nodes and edges:
n = G.number_of_nodes()
m = G.number_of_edges()

###################################################################################################
# Step 3.1: Density, transitivity and reciprocity                                                 #
###################################################################################################

# Density:
nx.density(G)

# Or equivalently:
m / (n * (n-1))

# Transitivity:
nx.transitivity(G)

# Reciprocity:
nx.reciprocity(G)

# The graph is fairly dense, as we saw from the plot.
# Reciprocity and transitivity are high as well.

###################################################################################################
# Step 3.2: Bridges and components                                                                #
###################################################################################################

# Most of the methods in this section are only implemented for undirected graphs.

# The graph is connected:
nx.is_connected(G_u)

# Number of connected components is thus one:
nx.number_connected_components(G_u)

# We can then use the following to get the components:
components = [conn_comp for conn_comp in nx.connected_components(G_u)]

# Our graph is fairly well connected, so there are no bridges:
nx.has_bridges(G_u)

# Otherwise we could use the following to find them:
bridges = [bridge for bridge in nx.bridges(G_u)]

###################################################################################################
# Step 3.3: Centrality                                                                            #
###################################################################################################

# The degrees:
G_u.degree()
G.in_degree()
G.out_degree()

# Degree centralities:
nx.degree_centrality(G_u)
nx.in_degree_centrality(G)
nx.out_degree_centrality(G)

# Betweenness and closeness:
nx.betweenness_centrality(G)
nx.closeness_centrality(G)

# Eigenvector centrality:
nx.eigenvector_centrality(G)

# Naturally, Trump has the highest degrees by the definition of the network. We took all the people he follows,
# that is, my definition he has an out-degree of n-1 = 47. All of the people he follows also follow him, so
# his in-degree is also 47. This is reflected in the degree centralities.

# We see that he also tops the other centrality measures:
np.argmax(list(nx.betweenness_centrality(G).values()))
np.argmax(list(nx.closeness_centrality(G).values()))
np.argmax(list(nx.eigenvector_centrality(G).values()))

# Notice that the centralities are returned as dictionaries, so we use the value() method to retrieve the values as list.
# Notice also that python uses the convention that the first element in a list is number 0.

# The second highest for the eigenvector centrality is vertex 46:
ind_second = np.argmax(list(nx.eigenvector_centrality(G).values())[1:])

# We can see that this is Trump Junior:
names_ids['names'][ind_second]

###################################################################################################
# Step 3.4: Spectral Clustering                                                                   #
###################################################################################################

# The normalised Laplacian matrix:
L = nx.normalized_laplacian_matrix(G_u).toarray()

# As the command returns a "sparse matrix object", we have to use the toarray() method to obtain a matrix we can work with in the standard way.

# To find the eigenvalues and eigenvectors of a symmetric matrix in python, we use:
np.linalg.eigh(L)

# Inspect the eigenvalues:
np.linalg.eigh(L)[0]

# It's hard to see how many communities are here. Also from the plot. We will experiment with k = 2 and 3.
k = 3

# The relevant eigenvectors:
U = np.linalg.eigh(L)[1][:,:k]

# Row normalisation:
row_norm = np.sqrt(np.sum( U ** 2, axis = 1))
N = 1 / row_norm
X = N[:,None] * U 

# When we create N, it becomes a [48 x 0] array (use np.shape(N) to check).
# This is how python thinks of a vector. To be able to use it in matrix multiplication,
# we need to make it a [48 x 1] dimensional array. This is done by writing N[:,None].
# You can check this with np.shape(N[:,None]).

# k-means++ clustering on the first k eigenvectors:
est = KMeans(n_clusters = k)
ind_sc = est.fit_predict(X)

# Plotting again:
fig, ax = plt.subplots(figsize = (10, 8))
nx.draw_networkx(G, lay, node_size = 150, width = 0.25, alpha = 0.75, edge_color = 'grey', node_color = ind_sc, with_labels = False, ax = ax)

labels = names_ids['names']
lab_lay = {key:(val + 0.05) for key,val in lay.items()}
nx.draw_networkx_labels(G, lab_lay, labels, font_size = 10, font_color = 'blue')

plt.show()

###################################################################################################
# Step 3.5: Modularity                                                                            #
###################################################################################################

# Modularity:
communities = nx.algorithms.community.greedy_modularity_communities(G_u)

# This returns a list of communities. For example, the members of community 2 are:
sorted(list(communities[1]))

# Creating an index vector for colouring:
ind_m = np.empty([n,1])

# Assigning the community numbers:
for i in range(len(communities)):
    ind_m[list(communities[i])] = i

# Plotting:
fig, ax = plt.subplots(figsize = (10, 8))
nx.draw_networkx(G, lay, node_size = 150, width = 0.25, alpha = 0.75, edge_color = 'grey', node_color = ind_m, with_labels = False, ax = ax)

labels = names_ids['names']
lab_lay = {key:(val + 0.05) for key,val in lay.items()}
nx.draw_networkx_labels(G, lab_lay, labels, font_size = 10, font_color = 'blue')

plt.show()