# set working directory 
setwd("~/BI-2020/Data-Science/Projects/Data-Science-Project/Data")

# import packages
pacman::p_load(ISLR, dplyr, ggplot2, tidyverse, GGally, corrplot, caret, 
               e1071, MASS, class, readxl, reshape2, openair)

##### weather data #####

# import dataset
weather = read_excel("HistoricalWeatherData2.xlsx")

# convert Date and Time to DateTime 
weather$Time = format(as.POSIXct(weather$Time, format = '%Y-%m-%d %H:%M:%S'), format = "%H:%M:%S")
weather$Datetime = as.POSIXct(paste(weather$Date, weather$Time), format="%Y-%m-%d %H:%M:%S")

weather %>% 
  str()

# reduce dataframe to include only DateTime measures and Values
weather = weather[, c(3,4,6)]


# transpose dataframe 
weather = dcast(weather, formula = Datetime~Measure, value.var = "Value",
                fun.aggregate = function(x) if(length(x) == 0) NA_real_ else sum(x, na.rm = TRUE))

str(weather)
# write to csv
# write.csv(x = weather, file = 'weather-by-hour2.csv')

##### Cafe data #####
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
df <- merge(x = date, y = cafe, by = "Datetime", all.x = TRUE)
df[is.na(df)] <- 0
str(df)

df <- merge(x = df, y = weather, by = "Datetime", all.x = TRUE)

##### sun data #####

# import dataset
sun = read_excel("SunData.xlsx")
sun = filter(sun, sun$`Station ID` == "Horsens")


# convert Date and Time to DateTime 
sun$Time = format(as.POSIXct(sun$Time, format = '%Y-%m-%d %H:%M:%S'), format = "%H:%M:%S")
sun$Datetime = as.POSIXct(paste(sun$Date, sun$Time), format="%Y-%m-%d %H:%M:%S")

sun %>% 
  str()

# reduce dataframe to include only DateTime measures and Values
sun = sun[, c(3,4,5,6)]

# transpose dataframe 
sun = dcast(sun, formula = Datetime~Measure, value.var = "Value",
            fun.aggregate = function(x) if(length(x) == 0) NA_real_ else sum(x, na.rm = TRUE))

# merge into df
df = merge(x=df, y=sun, by = "Datetime", all.x = TRUE)
