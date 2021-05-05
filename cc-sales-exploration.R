# set working directory 
setwd("~/BI-2020/Data-Science/Projects/Data-Science-Project")

# import packages
pacman::p_load(ISLR, dplyr, ggplot2, tidyverse, GGally, corrplot, caret, 
               e1071, MASS, class, readxl, reshape2, keras, scales, gridExtra,
               grid, zoo, lubridate)

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

# plot series 
p_cross <- ggplot(cafe, aes(Datetime, cross_cafe)) +
  geom_point() +
  ggtitle("hourly sales\n cross\n 2018-2021") +
  xlab("Date") + ylab("sales") +
  scale_x_datetime(labels=date_format ("%m-%y"))+
  theme(plot.title = element_text(lineheight=.8, face="bold",
                                  size = 20)) +
  theme(text = element_text(size=18))
p_cross

p_Aaland <- ggplot(cafe, aes(Datetime, Aaland)) +
  geom_point() +
  ggtitle("hourly sales\n Aaland\n 2018-2021") +
  xlab("Date") + ylab("sales") +
  scale_x_datetime(labels=date_format ("%m-%y"))+
  theme(plot.title = element_text(lineheight=.8, face="bold",
                                  size = 20)) +
  theme(text = element_text(size=18))
p_Aaland

p_ziggy <- ggplot(cafe, aes(Datetime, Ziggy)) +
  geom_point() +
  ggtitle("hourly sales\n Ziggy\n 2018-2021") +
  xlab("Date") + ylab("sales") +
  scale_x_datetime(labels=date_format ("%m-%y"))+
  theme(plot.title = element_text(lineheight=.8, face="bold",
                                  size = 20)) +
  theme(text = element_text(size=18))
p_ziggy
