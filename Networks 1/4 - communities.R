###################################################################################################
#                                                                                                 #
#                                   Community structure                                           #
#                                                                                                 #
###################################################################################################

###################################################################################################
# Preliminaries                                                                                   #
###################################################################################################

# Loading the necessary libraries:
library(sna)
library(LICORS)   # For k-means and related algorithms.

###################################################################################################
# Part 1: Setting up the network                                                                  #
###################################################################################################

# Set working directory to where you put the data "2b - campnet.csv" and "2c - campnet_attr.csv"
setwd("/home/stefan/Documents/teaching/applied-data-science/codes")

# Import the data:
campnet <- read.csv(file="2b - campnet.csv", header=T, row.names=1, as.is=T)

# Reading in the attributes:
campnet.attr <- read.csv(file="2c - campnet_attr.csv", header=T, as.is=T)

# For the purposes of this exercise, we will treat the network as undirected.

# Create the network object:
camp_net <- network(as.matrix(campnet), directed = F)

# Plotting and saving the coordinates:
layout_coordinates <- gplot(camp_net, mode="fruchtermanreingold", displaylabels=T, label.cex=0.5, label.col="blue", vertex.col=campnet.attr$Gender)

# Notice that there is evidence of a community structure in this plot. Based on visual inspection, we could informally argue for 3 communities.

###################################################################################################
# Part 2: Spectral clustering                                                                     #
###################################################################################################

# Number of vertices
n = network.size(camp_net)

# The adjacency matrix:
A = as.matrix.network(camp_net)

# The degrees:
degrees = rowSums(A)

# Equivalently:
degrees = degree(camp_net)/2

# The degree matrix:
D = diag(degrees)

# The unnormalised Laplacian:
L_u = D - A

# Notice that its rows and columns sum to zero:
rowSums(L_u)
colSums(L_u)

# The smallest eigenvalue is zero, but not the second as the graph is connected:
eigen(L_u)$val
plot(eigen(L_u)$val)

# Looking at the plot, we could argue that the first (smallest) 3 eigenvalues are closer to zero than the rest.
# This would suggest that we should consider k = 3 communities (but keep in mind that this is far from a rigorous argument).

# The inverted square root of the degree matrix:
D_isq = diag(1 / sqrt(degrees))

# Calculating the normalised Laplacian:
L = D_isq %*% (D - A) %*% D_isq

# Notice that we use %*% for matrix multiplication in R.

# The eigenvalues are now scaled differently, but the shape of the plot looks very similar:
eigen(L)$val
plot(eigen(L)$val)

# We set the number of communities to three based on the eigenplots:
k = 3

# Selecting the first k eigenvectors (actually, they are ordered last in R):
U = eigen(L)$vec[,(n-k+1):n]

# Normalise:
N <- diag(1/(sqrt(rowSums(U^2))) )
X <- N %*% U 

# Clustering: 
communities_sc = kmeanspp(X, k)$cluster

# k-means++ (kmeanspp) is a variant of k-means that has better performance guarantees.

# Plotting with community colours:
gplot(camp_net, coord=layout_coordinates, displaylabels=T, label.cex=0.5, label.col="blue", vertex.col=communities_sc)

# The communities are more or less what we could spot from looking at the network plot from before.

###################################################################################################
# Part 3: Modularity                                                                              #
###################################################################################################

# We load the igraph library:
library(igraph)

# igraph is a powerful library for analysing networks, which amongst other things
# has many community detection algorithms.

# We begin by setting up the graph for igraph:
graph = graph_from_adjacency_matrix(as.matrix(campnet), mode = "undirected")

# Communities from modularity method:
communities_m = membership(cluster_leading_eigen(graph))

# Plotting:
gplot(camp_net, coord=layout_coordinates, displaylabels=T, label.cex=0.5, label.col="blue", vertex.col=communities_m)

# We obtain the same communities as spectral clustering in this case.
# Notice that modularity also finds k for us, although this is not necessarily going to be the "correct" k.

###################################################################################################
# Part 4: Girvan-Newman                                                                           #
###################################################################################################

# Communities from modularity method:
communities_gn = membership(cluster_edge_betweenness(graph))

# Plotting:
gplot(camp_net, coord=layout_coordinates, displaylabels=T, label.cex=0.5, label.col="blue", vertex.col=communities_gn)

# We obtain the same communities as the methods in this case as well. This is not surprising as the structure is fairly obvious.

# As we saw in the slides, this method actually returns a sequence of communities. Every time we remove an edge, we can 
# potentially have a new community structure. What this particular implementation of the method does is to calculate
# a modularity based score for all of these community partitions, and return the one with the highest score.