###################################################################################################
#                                                                                                 #
#                                   Visualising networks                                          #
#                                                                                                 #
###################################################################################################

###################################################################################################
# Preliminaries                                                                                   #
###################################################################################################

# Loading the necessary libraries:
library(sna)

###################################################################################################
# Part 1: Importing the data                                                                      #
###################################################################################################

# Set working directory to where you put the data "2b - campnet.csv" and "2c - campnet_attr.csv"
setwd("/home/stefan/Documents/teaching/applied-data-science/codes")

# Import the data:
campnet <- read.csv(file="2b - campnet.csv", header=T, row.names=1, as.is=T)

# See the first few columns of the data:
head(campnet)

# The campnet dataset contains interactions among 18 people participating in a workshop (including 4 instructors). 
# An edge from vertex i to j indicates that person i listed person j as one of their top three interactors.

# Reading in the attributes:
campnet.attr <- read.csv(file="2c - campnet_attr.csv", header=T, as.is=T)
campnet.attr

# The attributes are:
# Gender: 1: female,      2: male
# Role:   1: participant, 2: instructor
# Combo:  1: female,      2: male,      3: instructor (all males)

# Set up the adjacency matrix and network:
A = as.matrix(campnet)
camp_net <- network(A)
summary(camp_net)

###################################################################################################
# Part 2: Visualising the network                                                                 #
###################################################################################################

# We can use the basic plot function to visualise the network:
plot(camp_net)

# sna also has the more flexible gplot function:
gplot(camp_net, mode="fruchtermanreingold", displaylabels=T, label.cex=0.5, label.col="blue")

# The "mode" parameter selects the network layout.
# "displaylabels" shows the vertex labels.
# label.cex = 0.5 sets the vertex label size to 0.5 of the default size.
# label.col = "blue" sets the color of all the labels to blue.

# For a list of layouts, see:
# https://www.rdocumentation.org/packages/sna/versions/2.5/topics/gplot.layout
# or page 86 of https://cran.r-project.org/web/packages/sna/sna.pdf.
# Try for example circle, eigen, kamadakawaii, mds and spring.

# Next we colour the vertices by gender:
gplot(camp_net, mode="fruchtermanreingold", displaylabels=T, label.cex=0.5, label.col="blue", vertex.col=campnet.attr$Gender)

# Notice that we have something of a community structure in this network.

# We can also ask R for specific colours for gender. First let's check out the colour names that R knows about:
colors()

# Create a vector of colours depending on gender:
vertex_colours <- ifelse(campnet.attr$Gender=="1","yellow","green")

# Then we plot:
gplot(camp_net, mode="fruchtermanreingold", displaylabels=T, label.cex=0.5, label.col="blue", vertex.col=vertex_colours)

# Notice that if we run the same command again, the vertex layout typically changes:
gplot(camp_net, mode="fruchtermanreingold", displaylabels=T, label.cex=0.5, label.col="blue", vertex.col=vertex_colours)

# This is normal, many layout algorithms return slightly different layouts upon repeated runs. However, suppose we want to 
# visualise the same network with different colourings. Then we want the vertices to be in the same place in all the figures.

# We can save the layout coordinates:
layout_coordinates <- gplot(camp_net, mode="fruchtermanreingold", displaylabels=T, label.cex=0.5, label.col="blue", vertex.col=vertex_colours)

# Then we can plot again using the same coordinates:
gplot(camp_net, displaylabels=T, label.cex=0.5, label.col="blue", vertex.col=campnet.attr$Role, coord = layout_coordinates, vertex.cex = 3)

# Examine the help file to find many more parameters controlling network visualisation: 
?sna::gplot 

# To save the plot to file, we first start a graphic device for PDFs, plot, and finally close the PDF device:
pdf("test.pdf")
gplot(camp_net, displaylabels=T, label.cex=0.5, label.col="blue", vertex.col=vertex_colours, coord=layout_coordinates)
dev.off()