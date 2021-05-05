# set working directory 
setwd("~/BI-2020/Data-Science/Projects/Data-Science-Project/Data")

# import packages
pacman::p_load(ISLR, dplyr, ggplot2, tidyverse, GGally, corrplot, caret, 
               e1071, MASS, class, readxl, reshape2)

##### process weather data #####

# import dataset
weather = read_excel("HistoricalWeatherData.xlsx")

# convert Date and Time to DateTime 
weather$Time = format(as.POSIXct(weather$Time, format = '%Y-%m-%d %H:%M:%S'), format = "%H:%M:%S")
weather$Datetime = as.POSIXct(paste(weather$Date, weather$Time), format="%Y-%m-%d %H:%M:%S")

weather %>% 
  str()

# reduce dataframe to include only DateTime measures and Values
weather = weather[, 3:5]


# transpose dataframe 
weather = dcast(weather, formula = Datetime~Measure, sum, value.var = "Value")

str(weather)
# write to csv
write.csv(x = weather, file = 'weather-by-hour.csv')


