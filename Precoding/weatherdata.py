import pandas as pd
import openpyxl
import fastf1 as ff1

###################### Loading Data 

# Load the Excel file
rawdata = pd.read_excel('weather_data.xlsx')  # Adjust the path if necessary
weather_data = rawdata[['Date time', 'Conditions']]
print(weather_data)


# Load the session data for the German Grand Prix, 2019 Race
race = ff1.get_session(2019, 'German Grand Prix', 'R')  # 'R' stands for Race
race.load()


# Extract laps data and filter for Verstappen
race_laps = race.laps
raw_ver_laps = race_laps[(race_laps['Driver'] == 'VER')]
ver_laps = raw_ver_laps[['Driver', 'LapNumber', 'LapStartDate']]
print(ver_laps)

print("Checking for NaN columns and deleting null values")
print(ver_laps['LapStartDate'].isnull().sum())
ver_laps = ver_laps.dropna(subset=['LapStartDate'])

##################### Merging dataframes

# Convert datetime columns to proper datetime format
ver_laps['LapStartDate'] = pd.to_datetime(ver_laps['LapStartDate'])
weather_data['Date time'] = pd.to_datetime(weather_data['Date time'], format='%m/%d/%Y %H:%M:%S')

# Sort both DataFrames by datetime to use merge_asof
ver_laps = ver_laps.sort_values('LapStartDate')
weather_data = weather_data.sort_values('Date time')

# Perform an asof merge, looking backward for the closest weather condition
lap_weather = pd.merge_asof(ver_laps, weather_data, left_on='LapStartDate', right_on='Date time', direction='backward')

# Drop the redundant 'Date time' column
lap_weather = lap_weather.drop(columns=['Date time'])
print(lap_weather)

