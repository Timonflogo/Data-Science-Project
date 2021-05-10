# set working directory 
setwd("~/BI-2020/Data-Science/Projects/Data-Science-Project/Data")

# import packages
pacman::p_load(ISLR, dplyr, ggplot2, tidyverse, GGally, corrplot, caret, 
               e1071, MASS, class, readxl, reshape2, keras, scales, gridExtra,
               grid, zoo, lubridate)

# import and inspect dataset
weather = read.csv("weather-by-hour.csv")
str(weather)

weather$Datetime = as.POSIXct(weather$Datetime, format = '%Y-%m-%d %H:%M:%S')
str(weather)

weather = weather[, -1]

# plot data
p_temp_dew <- ggplot(weather, aes(x=Datetime, y=temp_dew)) +
  geom_line() +
  xlab("")
p_temp_dew

p_humidity<- ggplot(weather, aes(x=Datetime, y=humidity)) +
  geom_line() +
  xlab("")
p_humidity

p_temp_dry <- ggplot(weather, aes(x=Datetime, y=temp_dry)) +
  geom_line() +
  xlab("")
p_temp_dry

p_wind_speed <- ggplot(weather, aes(x=Datetime, y=wind_speed)) +
  geom_line() +
  xlab("")
p_wind_speed

p_temp_mean_past1h <- ggplot(weather, aes(x=Datetime, y=temp_mean_past1h)) +
  geom_line() +
  xlab("")
p_temp_mean_past1h

p_pressure <- ggplot(weather, aes(x=Datetime, y=pressure)) +
  geom_line() +
  xlab("")
p_pressure

p_clouds <- ggplot(weather, aes(x=Datetime, y=cloud_cover)) +
  geom_line() +
  xlab("")
p_clouds


###### create facet plots for different time periods ######
weather$year <- year(weather$Datetime)
weather$month <- format(weather$Datetime, "%B")
# use factor instead to allow for levels
weather$month <- factor(weather$month, levels = c(
  'January','February','March',
  'April','May','June','July',
  'August','September','October',
  'November','December'))

# weather$day <- day(weather$Datetime)

# whole period
p_temp_dry_daily <- ggplot(weather, aes(Datetime, temp_dry)) +
  geom_point() +
  ggtitle("Daily Air Temperature\n Aarhus\n 2018-2021") +
  xlab("Date") + ylab("Temperature") +
  scale_x_datetime(labels=date_format ("%m-%y"))+
  theme(plot.title = element_text(lineheight=.8, face="bold",
                                  size = 20)) +
  theme(text = element_text(size=18))
p_temp_dry_daily

p_temp_mean_1h <- ggplot(weather, aes(Datetime, temp_mean_past1h)) +
  geom_point() +
  ggtitle("Daily mean Temperature\n Aarhus\n 2018-2021") +
  xlab("Date") + ylab("Temperature") +
  scale_x_datetime(labels=date_format ("%m-%y"))+
  theme(plot.title = element_text(lineheight=.8, face="bold",
                                  size = 20)) +
  theme(text = element_text(size=18))
p_temp_mean_1h

p_wind_speed <- ggplot(weather, aes(Datetime, wind_speed)) +
  geom_point() +
  ggtitle("Daily wind_speed\n Aarhus\n 2018-2021") +
  xlab("Date") + ylab("wind speed") +
  scale_x_datetime(labels=date_format ("%m-%y"))+
  theme(plot.title = element_text(lineheight=.8, face="bold",
                                  size = 20)) +
  theme(text = element_text(size=18))
p_wind_speed


p_humidity <- ggplot(weather, aes(Datetime, humidity)) +
  geom_point() +
  ggtitle("Daily wind_speed\n Aarhus\n 2018-2021") +
  xlab("Date") + ylab("HUmidity") +
  scale_x_datetime(labels=date_format ("%m-%y"))+
  theme(plot.title = element_text(lineheight=.8, face="bold",
                                  size = 20)) +
  theme(text = element_text(size=18))
p_humidity

# by year
p_temp_dry_daily <- ggplot(weather, aes(temp_dew, temp_dry)) +
  geom_point() +
  ggtitle("Daily Air Temperature\n Aarhus\n 2018-2021") +
  xlab("temp_dew") + ylab("temp_dry") +
  theme(plot.title = element_text(lineheight=.8, face="bold",
                                  size = 20)) +
  theme(text = element_text(size=18))
p_temp_dry_daily

p_temp_dry_daily + facet_wrap(~ month, nc=3)


