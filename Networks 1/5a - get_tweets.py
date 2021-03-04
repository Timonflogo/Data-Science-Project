"""
This script downloads recent tweets either based on a search query or from a 
specific user account and saves it to file.

Notice that you need to have a Twitter account and apply for credentials
for their API. You will easily find guides on how to do that on the internet.
Once you have your keys and secret, replace the XXX below to download tweets.

Notice also that Twitter limits what we can do with a free account. When harvesting
tweets by a search query, we can only get tweets from the last 7 days.

Regular expressions reminder:
    http://www.zytrax.com/tech/web/regex.htm
"""

###################################################################################################
# Preliminaries                                                                                   #
###################################################################################################

# Packages:
import re               # For regular expressions.
import tweepy as tw     # For harvesting Twitter data.

# Keys:
consumer_key        = 'XXX'
consumer_secret     = 'XXX'
access_token        = 'XXX'
access_token_secret = 'XXX'

# Authenticating and logging-on:
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit = True)

# Parameters:
num_tweets = 500

###################################################################################################
# Step 1: Downloading tweets                                                                      #
###################################################################################################

# Downloading data by query:
#query = '#trump'
#tweet_objects = tw.Cursor(api.search, q = query + ' -filter:retweets', lang = 'en', tweet_mode = 'extended').items(num_tweets)
#tweets = [tweet.full_text for tweet in tweet_objects]

# Downloading data by user:
user = 'CNN'
tweet_objects = tw.Cursor(api.user_timeline, screen_name = user, lang = 'en', tweet_mode = 'extended', exclude_replies = True, include_rts = False).items(num_tweets)
tweets = [tweet.full_text for tweet in tweet_objects]

###################################################################################################
# Step 2: Cleaning up the tweets                                                                  #
###################################################################################################

# Function for removing URLs from the tweets:
def remove_url(text):
    return re.sub("\w+:\/\/\S+", "", text)

# Function for removing punctuation (including hashtags) and extra whitespace:
def remove_punct(text):
    return ' '.join(re.sub("[^0-9A-Za-z \t]", "", text).split())

# Twitter replaces the ampersand (&) with "&amp;", ">" with "&gt;" and "<" "&lt;". Remove this and some other symbols:
clean_tweets_1 = [tweet.replace('&amp;', '').replace('&gt;', '').replace('&lt;', '').replace('&pm;', '').replace('&et;', '') for tweet in tweets]

# Cleaning URLs and punctuation:
clean_tweets_2 = [remove_punct(remove_url(tweet)) for tweet in clean_tweets_1]
    
# Removing capitalisation:
clean_tweets = [tweet.lower() for tweet in clean_tweets_2]

###################################################################################################
# Step 3: Saving the tweets                                                                       #
###################################################################################################

# Saving to file:
with open("1b - tweets.txt", "w") as f:
    for i in xrange(len(clean_tweets)):
        f.write(clean_tweets[i] + '\n')