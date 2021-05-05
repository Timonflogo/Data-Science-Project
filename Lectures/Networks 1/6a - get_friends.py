"""
A script that gets the ids and screen names of the friends of Donald Trump's friends 
on Twitter and saves them to file.

As before, you need to obtain the tokens and secrets by registering for a Twitter
developer account to get this script to run.
"""

###################################################################################################
# Preliminaries                                                                                   #
###################################################################################################

# Packages:
import tweepy as tw            # For harvesting Twitter data.

# Keys:
consumer_key        = 'XXX'
consumer_secret     = 'XXX'
access_token        = 'XXX'
access_token_secret = 'XXX'

# Authenticating and logging-on:
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit = True, wait_on_rate_limit_notify = True)

# Parameters:
user_name = 'realDonaldTrump'

###################################################################################################
# Step 1: Getting and saving the ids and names of Trump's friends                                 #
###################################################################################################

# Getting Trump's id:
data = api.get_user(user_name)
own_id = data.id

# Getting the id of his friends (that is, the people who he follows):
friends_ids = api.friends_ids(user_name)

# Getting the screen names of the ids (this takes a while to run as Twitter only allows us to look up 15 people every 15 minutes):
friends_names = [user.screen_name for user in api.lookup_users(user_ids = friends_ids)]

# Mapping the names to ids:
names = [user_name] + friends_names
ids   = [own_id] + friends_ids
names_ids = [','.join(map(str, x)) for x in zip(names, ids)]

# Saving to file:
with open("2b - names_ids.txt", "w") as f:
    for i in xrange(len(names_ids)):
        f.write(names_ids[i] + '\n')

###################################################################################################
# Step 2: Getting and saving the ids of the friends of the friends                                #
###################################################################################################

# Getting the ids of the friends of the friends:
friends_of_friends_ids_all = list()
for friend in friends_names:
    friends_of_friends_ids_all.append(api.friends_ids(friend))

# Removing ids that are not amongst Trump's friends:
friends_of_friends_ids = [[friend_of_friend for friend_of_friend in friend if friend_of_friend in ids] for friend in friends_of_friends_ids_all]
friends_of_friends_ids = [','.join(map(str,friend)) for friend in friends_of_friends_ids]
        
# Saving to file:
with open("2c - friends_of_friends_ids.txt", "w") as f:
    for i in xrange(len(friends_of_friends_ids)):
        f.write(friends_of_friends_ids[i] + '\n')