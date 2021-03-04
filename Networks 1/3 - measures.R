###################################################################################################
#                                                                                                 #
#                                     Network measures                                            #
#                                                                                                 #
###################################################################################################

###################################################################################################
# Preliminaries                                                                                   #
###################################################################################################
######  comment #####

# Loading the necessary libraries:
library(sna)

###################################################################################################
# Part 1: Setting up the network                                                                  #
###################################################################################################

# Set working directory to where you put the data "2b - campnet.csv" and "2c - campnet_attr.csv"
setwd("/home/stefan/Documents/teaching/applied-data-science/codes")

# Import the data:
campnet <- read.csv(file="2b - campnet.csv", header=T, row.names=1, as.is=T)

# Reading in the attributes:
campnet.attr <- read.csv(file="2c - campnet_attr.csv", header=T, as.is=T)

# Create the network object:
camp_net <- network(as.matrix(campnet))

# Plotting:
layout_coordinates <- gplot(camp_net, mode="fruchtermanreingold", displaylabels=T, label.cex=0.5, label.col="blue", vertex.col=campnet.attr$Gender)

###################################################################################################
# Part 2: Calculating basic network measures                                                      #
###################################################################################################

###################################################################################################
# Part 2.1: Geodesic distances                                                                    #
###################################################################################################

# The length of the shortest path for all pairs of nodes:
camp_dist <- geodist(camp_net)

# Notice that vertices that cannot reach each other have a distance of infinity (Inf):
camp_dist$gdist

# Replace "Inf" with the longest theoretically possible path length in this network:
n = network.size(camp_net)
camp_dist <- geodist(camp_net, inf.replace = n)
camp_dist$gdist

# The number of shortest path for all pairs of nodes:
camp_dist$counts  

###################################################################################################
# Part 2.2: Density                                                                               #
###################################################################################################

# Create network from example in slides:
m = matrix(0, 5, 5)
m[1,2:4] = 1
m[2,3] = 1
m[4,5] = 1
net_1 <- network (m, directed=F, loops=FALSE, bipartite=FALSE, matrix.type="adjacency")

# Density:
gden(net_1, mode = "graph")

# Notice that we use mode "digraph" for directed graphs (the default) and "graph" for undirected networks.

# We can also calculate it manually as:
network.edgecount(net_1) / (5 * 4) * 2

# For campnet:
gden(camp_net)

# Manually:
network.edgecount(camp_net) / (n * (n-1))

###################################################################################################
# Part 2.3: Transitivity                                                                          #
###################################################################################################

# Clustering coefficient for example from slides:
gtrans(net_1, mode = "graph")

# This gives some warning messages but the result is correct.

# For campnet:
gtrans(camp_net)

###################################################################################################
# Part 2.4: Reciprocity                                                                           #
###################################################################################################

# Create network from example in slides:
m = matrix(0, 4, 4)
m[1,c(2,4)] = 1
m[2,c(1,3)] = 1
net_2 <- network (m, directed=T, loops=FALSE, bipartite=FALSE, matrix.type="adjacency")

# The reciprocity:
grecip(net_2, measure = "edgewise")

# Notice that we need the edgewise option to match the slides, as the default calculates a slightly different measure.

# Reciprocity for campnet:
grecip(camp_net, measure = "edgewise") 

###################################################################################################
# Part 2.5: Components                                                                            #
###################################################################################################

# Number of components:
components(camp_net, connected = "strong") 
components(camp_net, connected = "weak")

# To see which vertices belong to which component, use:
camp_comp <- component.dist(camp_net, connected = "strong")
camp_comp$membership

# Size of each component:
camp_comp$csize

# Finding cutpoints:
cutpoints(camp_net, connected = "strong")
cutpoints(camp_net, connected = "weak")

# When looking at strong components, vertex 1 is a cutpoint. 
# Let's remove it (that is, all of its edges) and look at the components again:
camp_net_cut <- camp_net[-1,-1]                 # "-1" selects all except the first row/column.
components(camp_net_cut, connected="strong")

###################################################################################################
# Part 3: Calculating centrality measures                                                         #
###################################################################################################

###################################################################################################
# Part 3.1: Calculating centrality measures for an undirected graph                               #
###################################################################################################

# We begin by calculating the centrality measures treating the graph as undirected.

# Degree centrality:
degree(camp_net, gmode="graph")      

# Betweenness centrality:
betweenness(camp_net, gmode="graph") 

# Closeness centrality:
closeness(camp_net, gmode="graph")  

# Eigenvector centrality:
evcent(camp_net, gmode="graph")

# As you can see, the measures can differ quite the lot.

###################################################################################################
# Part 3.2: Calculating centrality measures for a directed graph                                  #
###################################################################################################

# When we calculate the directed degree centrality, we have several options.

# In-degree centrality:
degree(camp_net, gmode="digraph", cmode="indegree")

# Out-degree centrality:
degree(camp_net, gmode="digraph", cmode="outdegree")

# Total degree (Freeman) centrality:
degree(camp_net, gmode="digraph", cmode="freeman")   

# Plotting the network with the vertex size proportional to the degree centrality:
campnet.attr$DegreeCent = degree(camp_net, gmode="graph")      
gplot(camp_net, coord = layout_coordinates, displaylabels=T, label.cex=0.5, label.col="blue", vertex.col=campnet.attr$Gender, vertex.cex=campnet.attr$DegreeCent*0.35)

###################################################################################################
# Part 4: Calculating centralisation measures                                                     #
###################################################################################################

# Degree centralisation, network treated as symmetric:
centralization(camp_net, FUN=degree, mode="graph")

# In-degree and out-degree centralisation, directed network:
centralization(camp_net, FUN=degree, mode="digraph", cmode="indegree")
centralization(camp_net, FUN=degree, mode="digraph", cmode="outdegree")

# Notice that the out-degree centralisation is zero, as the out-degree centrality is the same for all vertices.

# Closeness and betweenness centralisation, graph treated as symmetric:
centralization(camp_net, FUN=betweenness, mode="graph")
centralization(camp_net, FUN=closeness, mode="graph")