#install.packages("twitteR")

#using the twitteR package ####
library(twitteR)


#From the twitter app (Keys and tokens)
api_key <- "insert_API_key" # API key 
api_secret <- "insert_API_secret_key" #API secret key 
token <- "insert_Access_token" #token 
token_secret <- "insert_Access_token_secret" #token secret


setup_twitter_oauth(api_key, api_secret, token, token_secret) # setup for accessing twitter using the information above

tweets <- searchTwitter('#fakenews AND [facebook OR #facebook]', n=1000, lang = "en") # the function searchTwitter search for tweets based on the specified parameters

tweets.df <-twListToDF(tweets) # creates a data frame with one row per tweet

tweetDF <- as.data.frame(tweets.df)




