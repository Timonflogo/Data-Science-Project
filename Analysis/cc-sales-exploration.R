# set working directory 
setwd("~/BI-2020/Data-Science/Projects/Data-Science-Project/Data")

# import packages
pacman::p_load(ISLR, dplyr, ggplot2, tidyverse, GGally, corrplot, caret, 
               e1071, MASS, class, readxl, reshape2, keras, scales, gridExtra,
               grid, zoo, lubridate, forecast, tseries)

# import data
cafe <- read_excel("cafedata_salesbyhour.xlsx")
cafe <- cafe[, c(2,5,1,4,3)]
str(cafe)

# combine date and time
cafe$Datetime <- as.character(paste(cafe$DateFormat, cafe$Hour))
cafe$Datetime <- as.POSIXct(cafe$Datetime, format = "%Y-%m-%d %H:%M")
str(cafe)

# get cafes as columns
cafe <- cafe[, c(1,4,5)]
cafe = dcast(cafe, formula = Datetime~Cafe, sum, value.var = "Rev_NoVAT")

# rename cafe cross 
# cafe %>% rename(cross_cafe = `Cafe Cross`)
names(cafe)[3] <- "cross_cafe"

# generate datetime for the period of the data and put into a new Dataframe
min(cafe$Datetime)
max(cafe$Datetime)

date <- as.data.frame(seq(ymd_h("2018-01-02-00"), ymd_h("2021-05-02-24"), by = "hours"))
names(date)[1] <- "Datetime"

# merge simulated Dataframe into cafe Dataframe and replace NAs with 0
cafe1 <- merge(x = date, y = cafe, by = "Datetime", all.x = TRUE)
cafe1[is.na(cafe1)] <- 0
str(cafe1)

# # plot series 
# p_cross <- ggplot(cafe1, aes(Datetime, cross_cafe)) +
#   geom_point() +
#   ggtitle("hourly sales\n cross\n 2018-2021") +
#   xlab("Date") + ylab("sales") +
#   scale_x_datetime(labels=date_format ("%m-%y"))+
#   theme(plot.title = element_text(lineheight=.8, face="bold",
#                                   size = 20)) +
#   theme(text = element_text(size=18))
# p_cross
# 
# p_Aaland <- ggplot(cafe1, aes(Datetime, Aaland)) +
#   geom_point() +
#   ggtitle("hourly sales\n Aaland\n 2018-2021") +
#   xlab("Date") + ylab("sales") +
#   scale_x_datetime(labels=date_format ("%m-%y"))+
#   theme(plot.title = element_text(lineheight=.8, face="bold",
#                                   size = 20)) +
#   theme(text = element_text(size=18))
# p_Aaland
# 
# p_ziggy <- ggplot(cafe1, aes(Datetime, Ziggy)) +
#   geom_point() +
#   ggtitle("hourly sales\n Ziggy\n 2018-2021") +
#   xlab("Date") + ylab("sales") +
#   scale_x_datetime(labels=date_format ("%m-%y"))+
#   theme(plot.title = element_text(lineheight=.8, face="bold",
#                                   size = 20)) +
#   theme(text = element_text(size=18))
# p_ziggy


# time series plots of sales data
firstHour <- 24*(as.Date("2018-1-2 00:00:00")-as.Date("2018-1-1 00:00:00"))

##### Cross #####
cc.ts <- ts(cafe1$cross_cafe, start = c(2018,firstHour), frequency = 24*365)
view(cc.ts)
tsdisplay(cc.ts) # two lockdowns are visible for cross cafe. 

# tests
adf.test(cc.ts) # stationary
acf(cc.ts, lag.max = 50)
pacf(cc.ts, lag.max = 50)

# ARIMA
fitcc.ts <- auto.arima(cc.ts) # does not work due to memory issues

##### Aaland #####
aa.ts <- ts(cafe1$Aaland, start = c(2018,firstHour), frequency = 24*365)
tsdisplay(aa.ts) # two lockdowns are visible for aaland

# tests
adf.test(aa.ts) # stationary
acf(aa.ts, lag.max = 50)
pacf(aa.ts, lag.max = 50)

##### Ziggy #####
zi.ts <- ts(cafe1$Ziggy, start = c(2018,firstHour), frequency = 24*365)
tsdisplay(zi.ts) # opened after the first lockdown

# tests
adf.test(zi.ts) # stationary
acf(zi.ts, lag.max = 50)
pacf(zi.ts, lag.max = 50)


