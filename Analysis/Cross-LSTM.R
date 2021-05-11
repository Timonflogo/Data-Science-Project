# LSTM for Cross cafe\
# set working directory
setwd("~/BI-2020/Data-Science/Projects/Data-Science-Project/Data")

# import libraries
pacman::p_load(ISLR, dplyr, ggplot2, tidyverse, GGally, corrplot, caret, 
               e1071, MASS, class, readxl, reshape2, keras, scales, gridExtra,
               grid, zoo, lubridate, forecast, tseries, timetk, tidyquant,
               tibbletime, forcats, glue, openair)

# sun_spots <- datasets::sunspot.month %>%
#   tk_tbl() %>%
#   mutate(index = as_date(index)) %>%
#   as_tbl_time(index = index)
# sun_spots

# import dataset 
cross <- read_csv("cross-cafe-final.csv")

# get rid of Index, Aaland, and Ziggy
cross <- cross[, -c(1,3,5)]

# isolate lockdown period
names(cross)[1] <- "date"
cross2020 <- selectByDate(
  cross,
  start = "2020-05-18",
  end = "2020-12-08 ",
)

# isolate predictor variables
cross2020 <- cross2020[, -c(3,5,8:11, 13:19)]

# Run regression to see how well our features explain sales
fit.reg <- lm(cross_cafe ~ . -date, data = cross2020)
summary(fit.reg)

# drop pressure as it has no effect on sales
cross2020 <- cross2020[, -c(5)]
str(cross2020)

# Run regression without pressure to see how well our features explain sales
fit.reg <- lm(cross_cafe ~ . -date, data = cross2020)
summary(fit.reg)

# inspect series for stationarity
# convert cross2020 to timetibble
cross2020 %>%
  tk_tbl() %>%
  as_tbl_time(index = date)

# plot acf for cross_cafe
# we are going to use 144 steps as we are trying to doe a 7 day forecast
max_lag <- 10*24
firstHour.2020<- 24*(as.Date("2020-5-18 00:00:00")-as.Date("2020-1-1 00:00:00"))

# plot 2020 data
cc.ts <- ts(cross2020$cross_cafe, start(2020,firstHour.2020), frequency = 24*365)
tsdisplay(cc.ts)
acf(cc.ts, lag.max = max_lag)

# we have very strong autocorrelation which is good for predicting sales. we are probably going for lag 144 as it is equivalent to the same hour 7 days in the past.

