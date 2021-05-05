"""
This script carries out a simple analysis of the tweets contained in "5b - tweets.txt"

We first look at the most common words and then a network of bi-grams (sequences of two words).

The package we use for the network analysis is networkx. See the documentation here:
https://networkx.github.io/documentation/stable/reference/index.html
"""

###################################################################################################
# Preliminaries                                                                                   #
###################################################################################################

# Packages:
import collections                     # A set of helpful tools for counting.
import itertools                       # A set of helpful iterative tools.                    
import matplotlib.pyplot as plt        # For plotting.
import nltk                            # The Natural Language ToolKit
from nltk.corpus import stopwords     
import networkx as nx                  # For working with networks.
import pandas as pd                    # For working with dataframes.

# Downloading list of stop words (only need to do that once):
#nltk.download('stopwords')

# Parameters:
num_most_common_words = 20
num_most_common_bigrams = 20

###################################################################################################
# Step 1: Loading in the tweets and cleaning up                                                   #
###################################################################################################

# Reading in the data:
with open('5b - tweets.txt', 'rb') as f:
    clean_tweets = f.readlines()

# Splitting into a list of lists of words:
tweets_split_words_1 = [[word.decode('utf-8') for word in tweet.split()] for tweet in clean_tweets]

# The decode method on changes the "bytes" to a string. This is just a technicality in python 3.

# Removing stop words:
stop_words = set(stopwords.words('english'))
tweets_split_words = [[word for word in tweet if word not in stop_words] for tweet in tweets_split_words_1]

# Removing collection words if necessary:
#collection_words = ['trump']
#tweets_split_words = [[word for word in tweet if word not in collection_words] for tweet in tweets_split_words] 

# Flattening list:
words = list(itertools.chain(*tweets_split_words))

###################################################################################################
# Step 2: Plotting the most common words                                                          #
###################################################################################################

# Counting words and returning the most common:
words_counter = collections.Counter(words)

# Setting up as a dataframe:
words_df = pd.DataFrame(words_counter.most_common(num_most_common_words), columns = ['words', 'count'])
words_df.head()

# Plotting a horizontal bar graph:
fig, ax = plt.subplots(figsize = (8,8))
words_df.sort_values(by = 'count').plot.barh(x = 'words', y = 'count', ax = ax, color = 'purple')
ax.set_title('Most common words')
plt.show()

###################################################################################################
# Step 3: Exploring the co-occurence of words through bigrams                                     #
###################################################################################################

# Finding bigrams and flattening them:
bigrams_unflattened = [list(nltk.bigrams(tweet)) for tweet in tweets_split_words]
bigrams = list(itertools.chain(*bigrams_unflattened))

# Counting:
bigram_counter = collections.Counter(bigrams)

# Creating dataframe:
bigram_df = pd.DataFrame(bigram_counter.most_common(num_most_common_bigrams), columns = ['bigram', 'count'])

# Add index to dataframe:
bigram_df = bigram_df.set_index('bigram')

# Convert to dictionary:
bigram_dict = bigram_df.T.to_dict('records')[0]

# Create a network:
G = nx.Graph()

# Adding edges based on our dictionary:
for key, val in bigram_dict.items():
    G.add_edge(key[0], key[1], weight = (val * 10))
    
# Setting up plot object and network layout:
fig, ax = plt.subplots(figsize = (10, 8))
lay = nx.spring_layout(G, k = 1)

# Plotting:
nx.draw_networkx(G, lay, font_size = 16, width = 3, edge_color = 'grey', node_color = 'purple', with_labels = False, ax = ax)

# Setting up labels:
for key, val in lay.items():
    x, y = val[0] + 0.025, val[1] + 0.025
    ax.text(x, y, s = key, horizontalalignment = 'center', fontsize = 10)

plt.show()