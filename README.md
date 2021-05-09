
# Data Science Project for Cross Cafe - Aarhus

  

## Objective:

Inducing a more data-driven decision making in a café environment.

  

Cafés are typically small businesses, most café owners, own only a few cafés with little knowledge and resources for inducing data-driven decisions. This project would help a particular café extract it’s data from various systems and collect it in a dashboard to easily investigate when making key changes in the business. The main goal of which is to help the café get a better overview of their operation while also providing future insights through forecasting that help them make critical decisions looking forward.

  

### Data sources:

- Data from the café

- Potential interviews or other qualitative input

- Review data from various websites

  

## Wind data observations:

  

### Available (almost) every hour:

Cloud_cover - (0-100%)

Humidity - (0-100%)

Precip_dur_past1h - (0-60 min)

Precip_past1h - (precip in mm)

Pressure - (some missing data / different methods of measurements)

Temp_dew - (degrees c - suppose it is a measure of humidity, but not sure)

Temp_dry - (degrees c - “air temperature”- some way of disregarding humidity’s effect on the temperature(?))

Temp_max_past1h - (degrees c)

Temp_mean_past1h - (degrees c)

Temp_min_past1h - (degrees c)

Wind_dir - (in degrees (0-360) - maybe transform to categorical variable)

Wind_max_per10min_past1h - (m/s)

Wind_speed - (m/s)

Wind_speed_past1h - (m/s)

   <br />
  

### Available more than once a day, but not hourly:

Weather (no idea to be honest, values go from 100-185)

Temp_max_past_12h (degrees c, measured at 6:00 and 18:00)

Temp_min_past 12h (degrees c, measured at 6:00 and 18:00)

Wind_min_past1h (m/s)

  <br />

### DMI weather descriptions
  
temp_dry	°C	Present air temperature measured 2 m over terrain	10 min	X	X
temp_dew	°C	Present dew point temperature measured 2 m over terrain	10 min	X	X
temp_mean_past1h	°C	Latest hour's mean air temperature measured 2 m over terrain	Hourly	X	X
temp_max_past1h	°C	Latest hour's maximum air temperature measured 2 m over terrain	Hourly	X	X
temp_min_past1h	°C	Latest hour's minimum air temperature measured 2 m over terrain	Hourly	X	X
temp_max_past12h	°C	Last 12 hours maximum air temperature measured 2 m above ground. Measured at 0600 and 1800 UTC.	Twice a day	X	X
temp_min_past12h	°C	Last 12 hours minimum air temperature measured 2 m above ground. Measured at 0600 and 1800 UTC.	Twice a day	X	X
temp_grass	°C	Present air temperature measured at grass height (5-20 cm over terrain)	10 min	X	
temp_grass_max_past1h	°C	Latest hour's maximum air temperature measured at grass height (5-20 cm over terrain)	Hourly	X	
temp_grass_mean_past1h	°C	Latest hour's mean air temperature measured at grass height (5-20 cm over terrain)	Hourly	X	
temp_grass_min_past1h	°C	Latest hour's minimum air temperature measured at grass height (5-20 cm over terrain)	Hourly	X	
temp_soil	°C	Present temperature measured at a depth of 10 cm	10 min	X	
temp_soil_max_past1h	°C	Latest hour's maximum temperature measured at a depth of 10 cm	Hourly	X	
temp_soil_mean_past1h	°C	Latest hour's mean temperature measured at a depth of 10 cm	Hourly	X	
temp_soil_min_past1h	°C	Latest hour's minimum temperature measured at a depth of 10 cm	Hourly	X	
humidity	%	Present relative humidity measured 2 m over terrain	10 min	X	X
humidity_past1h	%	Latest hour's mean for relative humidity measured 2 m over terrain	Hourly	X	X
pressure	hPa	Atmospheric pressure at station level	10 min	X	X
pressure_at_sea	hPa	Atmospheric pressure reduced to mean sea level	10 min	X	X
wind_dir	degree	Latest 10 minutes' mean wind direction measured 10 m over terrain. 0 means calm (see Codes (metObs))	10 min	X	X
wind_dir_past1h	degree	Latest hour's mean wind direction measured 10 m over terrain	Hourly	X	X
wind_speed	m/s	Latest 10 minutes' mean wind speed measured 10 m over terrain	10 min	X	X
wind_speed_past1h	m/s	Latest hour's mean wind speed measured 10 m over terrain	Hourly	X	X
wind_gust_always_past1h	m/s	Latest hour's highest 3 seconds mean wind speed measured 10 m over terrain	Hourly	X	X
wind_max	m/s	Latest 10 minutes' highest 3 seconds mean wind speed measured 10 m over terrain	10 min	X	X
wind_min_past1h	m/s	Latest hours lowest 3 second mean wind speed measured 10 m over terrain	Hourly	X	X
wind_min	m/s	Latest 10 minutes' lowest 3 seconds mean wind speed measured 10 m over terrain	10 min	X	X
wind_max_per10min_past1h	m/s	Maximum 10-minute average wind speed in the one hour period preceding the time of observation	Hourly	X	X
precip_past1h	kg/m²	Accumulated precipitation in the latest hour or the code -0,1, which means "traces of precipitation, less than 0.1 kg/m²". kg/m² is equivalent to mm. (see Codes (metObs))	Hourly	X	X
precip_past10min	kg/m²	Accumulated precipitation in the latest 10 minutes. kg/m² is equivalent to mm.	10 min	X	X
precip_past1min	kg/m²	
Accumulated precipitation in the latest minute. kg/m² is equivalent to mm.

Data is sent to from the rain gauge to the DMI every 10 minutes, if it has rained within the 10 minute period. No data is sent, if it hasn't rained within the 10 minute period.

Therefore, data will be null, if it hasn't rained within the 10 minute period. If, however, it has rained within the 10 minute period, the minutes, where it has rained, will show, how much it has rained during each of the minutes in question, whereas the minutes, where it didn't rain will be shown as '0'. 

10 min	X	
precip_past24h*	kg/m²	Accumulated precipitation in the latest 24 hours or the code -0,1, which means "traces of precipitation, less than 0.1 kg/m²". kg/m² is equivalent to mm. (see Codes (metObs))	Daily	
X
precip_dur_past10min	minutes	Number of minutes with precipitation in the latest 10 minutes	10 min	X	
precip_dur_past1h	minutes	Number of minutes with precipitation in the latest hour	Hourly	X	
snow_depth_man	cm	Snow depth (measured manually) or the code -1, which means "less than 0.5 cm" (see Codes (metObs))	Daily	X	
snow_cover_man	enum	Snow cover (measured manually), specified as quarters of the earth covered 	Daily	X	
visibility	m	Present visibility	10 min	X	X
visib_mean_last10min	m	Latest 10 minutes' mean visibility	10 min	X	X
cloud_cover	%	Total cloud cover or code (see Codes (metObs))	10 min	X	X
cloud_height	m	Height to the lowest clouds	10 min	X	X
weather	enum	Present weather (see Codes (metObs))	10 min	X	X
radia_glob	W/m²	Latest 10 minutes global radiation mean intensity	10 min	X	X
radia_glob_past1h	W/m²	Mean intensity of global radiation in the latest hour	10 min	X	X
sun_last10min_glob	minutes	Number of minutes with sunshine the latest 10 minutes	10 min	X	
sun_last1h_glob	minutes	Number of minutes with sunshine the latest hour	Hourly	X	
leav_hum_dur_past10min	minutes	Number of minutes with leaf moisture the latest 10 minutes	10 min	X	
leav_hum_dur_past1h	minutes	Number of minutes with leaf moisture the latest hour	Hourly	X	
  

## written by:

Timon Florian Godt, Morten Hamburger, Daniel Bolander, Piratheban Rajasekaran 

