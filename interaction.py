import math
import fastf1 as ff1
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.path import Path
from plotly.subplots import make_subplots
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import nearest_points
from dash import Dash, dcc, html, Input, Output, State

# #########################################################
# Things to do:
# - Clean up legend (use showlegend=false)
# - continuous weather and race event icons
# - decide on colour scheme
# - race track plotting issues
# - fix interaction (done!)
#########################################################

'''
Constants that define how the code runs.
Change this if you'd like to see a different course, driver or lap.
'''
YEAR = 2019
GRAND_PRIX = 'German Grand Prix'
SESSION_TYPE = 'R' 
DRIVER = 'MAG'
SELECTED_LAPS = [4,16,50, 55]

'''
Constants that change the event hierarchy or visuals used in the charts.
'''
### Event Hierarchy classified based on seriousness ###
EVENT_HIERARCHY = {
    1: 1,  # All Clear
    2: 2,  # Yellow Flag
    6: 3,  # Virtual Safety Car deployed
    7: 3,  # Virtual Safety Car ending
    4: 4,  # Safety Car
    5: 5   # Red Flag
}

### Colours used for each compound type ###
TIRE_COLOUR = {
    'soft': '#F23838',        
    'medium': '#F2CB05',      
    'hard': '#FFFFFF',        
    'intermediate': '#5BA004',
    'wet': '#171695'          
}

### Shapes used for each compound type ###
TIRE_SHAPE = {
    'soft': 'circle',
    'medium': 'square',
    'hard': 'triangle-up',
    'intermediate': 'diamond',
    'wet': 'x'
}

### Lap event emojis ###
EVENT_EMOJIS = {
    1: '',           # All Clear (no emoji)
    2: '⚠️',         # Yellow Flag
    3: '🚗',         # Virtual Safety Car
    4: '🚓',         # Safety Car
    5: '🚩'          # Red Flag
}

'''
Load session, laps, race data and weather data and set position variables.
'''
race = ff1.get_session(YEAR, GRAND_PRIX, SESSION_TYPE)
race.load(weather = True)
rawlaps = race.laps
weather_data = race.weather_data
driver_laps = race.laps.pick_driver(DRIVER)
selected_laps = driver_laps.pick_laps(SELECTED_LAPS)
position_data_100 = []  # List to store position data for each lap at 100Hz
position_data_orig = [] # List to store position data for each lap at original frequency
x_coords_original = []
y_coords_original = []

# new weather data
rawdata = pd.read_excel('weather_data.xlsx')  # Adjust the path if necessary
weather_data = rawdata[['Date time', 'Conditions']]

##################### Merging dataframes - add weather conditions to laps
ver_laps = rawlaps.dropna(subset=['LapStartDate'])

# Convert datetime columns to proper datetime format
ver_laps['LapStartDate'] = pd.to_datetime(ver_laps['LapStartDate'])
weather_data['Date time'] = pd.to_datetime(weather_data['Date time'], format='%m/%d/%Y %H:%M:%S')

# Sort both DataFrames by datetime to use merge_asof
ver_laps = ver_laps.sort_values('LapStartDate')
weather_data = weather_data.sort_values('Date time')

# Perform an asof merge, looking backward for the closest weather condition
laps = pd.merge_asof(ver_laps, weather_data, left_on='LapStartDate', right_on='Date time', direction='backward')

# Drop the redundant 'Date time' column
laps = laps.drop(columns=['Date time'])

'''
Load CSVs about the track.
'''
track_data_url = "https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/tracks/Hockenheim.csv"
df = pd.read_csv(track_data_url)
raceline_url = 'https://github.com/TUMFTM/racetrack-database/raw/e59595d1f3573b30d1ded6a08984935b957688e0/racelines/Hockenheim.csv'
raceline_data = pd.read_csv(raceline_url, comment='#', header=None)

'''
Method to gather telemetry, and updates position and coordinate data.
Updates positions and returns telemetry variables.
'''
def get_telemetry_and_positions():
    for lap_number in SELECTED_LAPS:
        lap_data = selected_laps[selected_laps['LapNumber'] == lap_number] # Filter data for current lap

        # Get telemetry data at both frequencies
        telemetry_100 = lap_data.get_telemetry(frequency=100)
        telemetry_original = lap_data.get_telemetry(frequency='original')

        # Combine them into a DataFrame
        position_data_100.append(pd.DataFrame({
            'X': telemetry_100['X']/10, # Adjust from 1/10m -> 1m scale
            'Y': telemetry_100['Y']/10, # Adjust from 1/10m -> 1m scale
            'Lap': lap_number,
            'Distance': telemetry_100['Distance'],
            'Time': telemetry_100['Date']  # Add timestamps to check the data frequency
        }))

        # Combine them into a DataFrame
        position_data_orig.append(pd.DataFrame({
            'X': telemetry_original['X']/10, # Adjust from 1/10m -> 1m scale
            'Y': telemetry_original['Y']/10, # Adjust from 1/10m -> 1m scale
            'Lap': lap_number,
            'Distance': telemetry_original['Distance'],
            'Time': telemetry_original['Date']  # Add timestamps to check the data frequency
        }))

    # Check if positional telemetry is available (X, Y data)
    if 'X' in telemetry_original.columns and 'Y' in telemetry_original.columns:
        # Extract the x and y coordinates
        x_coords = telemetry_original['X']/10 # Adjust from 1/10m -> 1m scale
        y_coords = telemetry_original['Y']/10 # Adjust from 1/10m -> 1m scale


    # Check if positional telemetry is available (X, Y data)
    if 'X' in telemetry_original.columns and 'Y' in telemetry_original.columns:
        # Extract the x and y coordinates
        x_coords_original = telemetry_original['X']/10 # Adjust from 1/10m -> 1m scale
        y_coords_original = telemetry_original['Y']/10 # Adjust from 1/10m -> 1m scale

        # Combine them into a DataFrame
        position_data = pd.DataFrame({
            'X': x_coords,
            'Y': y_coords,
            'Time': telemetry_original['Date']  # Add timestamps to check the data frequency
        })

        # Combine them into a DataFrame
        position_data_original = pd.DataFrame({
            'X': x_coords_original,
            'Y': y_coords_original,
            'Time': telemetry_original['Date']  # Add timestamps to check the data frequency
        })

        ## unused code
        # # Calculate the time difference between consecutive telemetry points
        # time_diff = position_data['Time'].diff().dt.total_seconds()

        # # Calculate the time difference between consecutive telemetry points
        # time_diff_original = position_data_original['Time'].diff().dt.total_seconds()

        return telemetry_100, telemetry_original

'''
Method to calculate left and right boundaries of track.
Returns boundaries in the form of coordinates.
'''
def calculate_boundaries():
    x = df['# x_m'].values  # Adjust based on the actual column name
    y = df['y_m'].values    # Adjust based on the actual column name
    track_width_right = df['w_tr_right_m'].values
    track_width_left = df['w_tr_left_m'].values

    # Calculate the direction vectors (tangent vectors) between consecutive points
    dx = np.diff(x)
    dy = np.diff(y)

    # Calculate the magnitude of the tangent vectors
    tangent_norm = np.sqrt(dx**2 + dy**2)

    # Normalize the tangent vectors to get unit vectors
    tangent_x = dx / tangent_norm
    tangent_y = dy / tangent_norm

    # Calculate the normal vectors (perpendicular to tangent)
    normal_x = -tangent_y
    normal_y = tangent_x

    # Calculate the left and right boundaries by offsetting along the normal vectors
    x_left = x[:-1] + normal_x * track_width_left[:-1]
    y_left = y[:-1] + normal_y * track_width_left[:-1]
    x_right = x[:-1] - normal_x * track_width_right[:-1]
    y_right = y[:-1] - normal_y * track_width_right[:-1]

    # Ensure the right boundary is closed
    x_right = np.append(x_right, x_right[0])
    y_right = np.append(y_right, y_right[0])

    # Ensure the left boundary is closed
    x_left = np.append(x_left, x_left[0])
    y_left = np.append(y_left, y_left[0])

    return x_right, y_right, x_left, y_left

'''
Calculate and filter sector times.
Returns sector_time_df and sector_time_long
'''
def sector_time_calculation():
    sector_times_df = laps[['Driver', 'LapNumber', 'Sector1Time', 'Sector2Time', 'Sector3Time', 'LapTime', 'Compound', 'Conditions']].copy()

    # Update Sector1Time for LapNumber == 1 using LapTime - Sector2Time - Sector3Time
    condition = sector_times_df['LapNumber'] == 1
    sector_times_df.loc[condition, 'Sector1Time'] = sector_times_df.loc[condition, 'LapTime'] - \
                                                    sector_times_df.loc[condition, 'Sector2Time'] - \
                                                    sector_times_df.loc[condition, 'Sector3Time']

    # For some reason there where no information for this observation, therefore it was calculated.
    condition2 = (sector_times_df['LapNumber'] == 27.0) & (sector_times_df['Driver'] == 'GRO')
    sector_times_df.loc[condition2, 'LapTime'] = sector_times_df.loc[condition2, 'Sector3Time'] + \
                                                    sector_times_df.loc[condition2, 'Sector1Time'] + \
                                                    sector_times_df.loc[condition2, 'Sector2Time']


    condition3 = (sector_times_df['LapNumber'] == 29.0) & (sector_times_df['Driver'] == 'HAM')
    sector_times_df.loc[condition3, 'LapTime'] = sector_times_df.loc[condition3, 'Sector3Time'] + \
                                                    sector_times_df.loc[condition3, 'Sector1Time'] + \
                                                    sector_times_df.loc[condition3, 'Sector2Time']

    condition4 = (sector_times_df['LapNumber'] == 30.0) & (sector_times_df['Driver'] == 'HAM')
    sector_times_df.loc[condition4, 'LapTime'] = sector_times_df.loc[condition4, 'Sector3Time'] + \
                                                    sector_times_df.loc[condition4, 'Sector1Time'] + \
                                                    sector_times_df.loc[condition4, 'Sector2Time']

    condition5 = (sector_times_df['LapNumber'] == 26.0) & (sector_times_df['Driver'] == 'STR')
    sector_times_df.loc[condition5, 'LapTime'] = sector_times_df.loc[condition5, 'Sector3Time'] + \
                                                    sector_times_df.loc[condition5, 'Sector1Time'] + \
                                                    sector_times_df.loc[condition5, 'Sector2Time']
    
    # Create dummy columns to indicate the fastest times for each sector and lap time
    sector_times_df['FastestSector1'] = (sector_times_df['Sector1Time'] == sector_times_df.groupby('Driver')['Sector1Time'].transform('min')).astype(int)
    sector_times_df['FastestSector2'] = (sector_times_df['Sector2Time'] == sector_times_df.groupby('Driver')['Sector2Time'].transform('min')).astype(int)
    sector_times_df['FastestSector3'] = (sector_times_df['Sector3Time'] == sector_times_df.groupby('Driver')['Sector3Time'].transform('min')).astype(int)
    sector_times_df['FastestLap'] = (sector_times_df['LapTime'] == sector_times_df.groupby('Driver')['LapTime'].transform('min')).astype(int)

    # Sort the data by Driver and LapNumber for better readability
    sector_times_df = sector_times_df.sort_values(by=['Driver', 'LapNumber']).reset_index(drop=True)
    sector_times_df = sector_times_df.fillna(0)

    # Find rows with missing data (NaN values)
    nat_rows = sector_times_df[
        sector_times_df[['Sector1Time', 'Sector2Time', 'Sector3Time', 'LapTime', 'Compound', 'Conditions']].isna().any(axis = 1)
    ]

    sector_times_df['Sector1Time'] = pd.to_timedelta(sector_times_df['Sector1Time'])
    sector_times_df['Sector2Time'] = pd.to_timedelta(sector_times_df['Sector2Time'])
    sector_times_df['Sector3Time'] = pd.to_timedelta(sector_times_df['Sector3Time'])
    sector_times_df['LapTime'] = pd.to_timedelta(sector_times_df['LapTime'])

    # convert to seconds for plots
    sector_times_df['Sector1Time'] = sector_times_df['Sector1Time'].dt.total_seconds()
    sector_times_df['Sector2Time'] = sector_times_df['Sector2Time'].dt.total_seconds()
    sector_times_df['Sector3Time'] = sector_times_df['Sector3Time'].dt.total_seconds()
    sector_times_df['LapTime'] = sector_times_df['LapTime'].dt.total_seconds()

    sector_times_long = sector_times_df.melt(id_vars = ['Driver', 'LapNumber'], 
                                            value_vars = ['Sector1Time', 'Sector2Time', 'Sector3Time'], 
                                            var_name = 'Sector', value_name = 'Time')
    return sector_times_df, sector_times_long

'''
Approximates the lap where rainfall changes based on the minute where it is recorded. 
Only considers the minutes where the race is still ongoing.
Returns an array of laps where Rainfall has changed.
'''
# def estimate_weather_changes(change_indexes):
    ### Calculate total number of minutes race takes ###
#    max_laps_driver = laps.groupby('Driver')['LapNumber'].max().idxmax()
#    max_laps_driver_laps = laps[laps['Driver'] == max_laps_driver]
#    total_race_time_seconds = max_laps_driver_laps['LapTime'].fillna(pd.Timedelta(0)).sum().total_seconds()
#    total_race_time_minutes = math.ceil(total_race_time_seconds / 60)
#    max_laps_driver_laps = max_laps_driver_laps.copy()  
#    max_laps_driver_laps['CumulativeLapTime'] = max_laps_driver_laps['LapTime'].cumsum().dt.total_seconds()

    ### Track minutes where changes in Rainfall happens ###
#    for i in range(1, total_race_time_minutes):
#        if weather_data['Rainfall'][i] != weather_data['Rainfall'][i - 1]:
#            change_indexes.append(i)

    ### Approximate the minute to the closest lap where Rainfall changed ###
#    approximate_laps = []
#   for minute in change_indexes:
#        target_time_seconds = minute * 60 
#       lap_data = max_laps_driver_laps[max_laps_driver_laps['CumulativeLapTime'] >= target_time_seconds].iloc[0]
#        lap_number = lap_data['LapNumber']
#        if lap_number > 1:
#            previous_lap_time = max_laps_driver_laps[max_laps_driver_laps['LapNumber'] == lap_number - 1]['CumulativeLapTime'].values[0]
#        else:
#            previous_lap_time = 0 
        
#        lap_fraction = (target_time_seconds - previous_lap_time) / (lap_data['CumulativeLapTime'] - previous_lap_time)
#        exact_lap = lap_number - 1 + lap_fraction
        
#        approximate_laps.append(round(exact_lap, 1))

#    return approximate_laps

change_laps = []
previous_condition = None  # To keep track of the previous condition

# Identify the laps where the weather changes
for i, row in laps.iterrows():
    current_condition = row['Conditions']
    
    # If weather condition changes (or it's the first row), track the lap number
    if previous_condition is None or current_condition != previous_condition:
        change_laps.append(row['LapNumber'])
    
    # Update the previous condition for the next iteration
    previous_condition = current_condition

def get_weather_emoji(condition):
    if condition == 'Rain':
        return '🌧️'
    elif condition == 'Partially cloudy':
        return '⛅'
    elif condition == 'Clear':
        return '☀️'
    elif condition == 'Overcast':
        return '☁️'
    else:
        return '❓'

# Use the get_weather_emoji function to generate emojis for the laps where weather changes
weather_emojis = [get_weather_emoji(laps.loc[laps['LapNumber'] == lap, 'Conditions'].values[0]) for lap in change_laps]

'''
Process events that happened per lap in the race.
Classifies them accordingly based on EVENT_HIERARCHY.
Returns the lap_events
'''
def process_lap_events():
    lap_events = laps[(laps['Driver'] == DRIVER)]
    lap_events['TrackStatusHierarchy'] = lap_events['TrackStatus'].apply(lambda status: EVENT_HIERARCHY.get(int(str(status)[0]), float('inf')))
    lap_events = lap_events.reset_index(drop=True)
    lap_events.index += 1
    lap_events = lap_events.iloc[:-1] # remove last lap
    
    return lap_events

'''
Draw legend.
'''
def draw_legend(fig):
    for compound, color in TIRE_COLOUR.items():
        fig.add_trace(
            go.Scatter(
                x=[None], 
                y=[None],  
                mode='markers',
                marker=dict(size=10, color=color),
                name=f'{compound.capitalize()}',
                showlegend=True,
                legendgroup='compounds' 
            ),
            row=1, col=1
    )
    
    return fig

'''
Draw Plot 1, the scatterplot of Lap Time and Events.
'''
def draw_scatterplot(fig):
    lap_time_data = sector_times_df[(sector_times_df['Driver'] == DRIVER) & (sector_times_df['LapNumber'] != 65)]

    # Normalize LapTime values to the range [0, 1]
    lap_times = lap_time_data['LapTime']
    normalized_lap_times = (lap_times - lap_times.min()) / (lap_times.max() - lap_times.min())

    # Map normalized values to the opacity range [0.7, 1.0]
    opacity_values = 1.0 - (0.3 * normalized_lap_times)  # 0.3 is the range between 1.0 and 0.7
    fig.add_trace(go.Scatter(
        x = lap_time_data['LapNumber'],
        y = lap_time_data['LapTime'],
        mode = 'markers',
        name = '<extra></extra>', 
        marker = dict(
            size = 8,
            color = lap_time_data['Compound'].map(lambda x: TIRE_COLOUR.get(x.lower(), 'black')),
            symbol = lap_time_data['Compound'].map(lambda  x: TIRE_SHAPE.get(x.lower(), 'cross')),
            opacity=opacity_values  # Apply calculated opacity
        ),
        hovertemplate = 'Lap: %{x}<br>Lap Time: %{y:.2f} seconds<br>Tire Compound: %{text}<br>Weather Condition:%{hovertext}',
        text = lap_time_data['Compound'],
        hovertext = lap_time_data['Conditions'],
        showlegend = False
    ), row = 1, col = 1)

    ### Plot 1: Scatterplot Conditions ### 
    ## Rain
    fig.add_trace(
        go.Scatter(
            x = change_laps,# approximate_weather_change_laps,  
            y = [190] * len(change_laps), # approximate_weather_change_laps),  plot at y level
            mode = 'text',  
            text = weather_emojis,
            textposition = 'middle center',
            name = 'Rainfall Change Points',
            showlegend = False,
            hoverinfo='none',
            textfont=dict(
                size=25 
            )
        ),
        row = 1, col= 1
    )

    ### Events
    fig.add_trace(
        go.Scatter(
            x = lap_events.index,  
            y = [180] * len(lap_events),  # plot at y level
            mode = 'text',  
            text = lap_event_emojis,
            textposition = 'middle center',
            name = 'Lap Events',
            showlegend = False,
            hoverinfo='none',
            textfont=dict(
                size=25 
            )
        ),
        row = 1, col= 1
    )

    ### Update Axes
    fig.update_xaxes(range=[-1, 70], title_text = "Lap Number", row = 1, col = 1)
    fig.update_yaxes(title_text = "Lap Times (s)", range = [sector_times_df['LapTime'].min() * 0.8, sector_times_df['LapTime'].max() * 1.2], row = 1, col = 1)

    return fig

'''
Draws Plot 2, the racetrack.
'''
def draw_racetrack(fig):
    lap_time_data = sector_times_df[(sector_times_df['Driver'] == DRIVER) & (sector_times_df['LapNumber'] != 65)]

    offset_x = 71
    offset_y = 198

    x_left_shifted = x_left - offset_x
    y_left_shifted = y_left - offset_y
    x_right_shifted = x_right - offset_x
    y_right_shifted = y_right - offset_y

    # Combine x and y coordinates into polygon points for left and right boundaries
    right_boundary_points = np.column_stack((x_right_shifted, y_right_shifted))
    left_boundary_points = np.column_stack((x_left_shifted, y_left_shifted))

    inner_boundary = Polygon(zip(x_left_shifted, y_left_shifted))
    outer_boundary = Polygon(zip(x_right_shifted, y_right_shifted))

    for i in range(len(position_data_100)):
        # Create a mask for valid points (Check if points are within outer but not inner boundary)
        mask_100 = position_data_100[i].apply(lambda row: inner_boundary.contains(Point(row['X'], row['Y'])) and not outer_boundary.contains(Point(row['X'], row['Y'])), axis=1)
        mask_orig = position_data_orig[i].apply(lambda row: inner_boundary.contains(Point(row['X'], row['Y'])) and not outer_boundary.contains(Point(row['X'], row['Y'])), axis=1)

        # Update the DataFrames with filtered data
        position_data_100[i] = position_data_100[i][mask_100]
        position_data_orig[i] = position_data_orig[i][mask_orig]


    # Define different marker styles
    marker_styles = ['circle', 'x', 'square', 'diamond', 'cross', 'triangle-up', 'triangle-down']

    # Loop through each selected lap
    for i, lap_number in enumerate(SELECTED_LAPS):
        # Access the compound for the specific lap and map it to the color
        compound_on_lap = lap_time_data['Compound'].values[lap_number]  # Get the compound name (e.g., 'soft')
        color_for_lap = TIRE_COLOUR.get(compound_on_lap.lower(), 'black')  # Map to color
        
        # Line trace
        fig.add_trace(go.Scatter(
            x=position_data_100[i]['X'],
            y=position_data_100[i]['Y'],
            mode='lines',
            line=dict(color=color_for_lap),
            name=f'Lap {lap_number}',
            customdata=position_data_100[i]['Distance'],
            hovertemplate='Distance: %{customdata:.0f}'
        ), row=1, col=2)
        
        # Marker trace with different marker styles
        fig.add_trace(go.Scatter(
            x=position_data_orig[i]['X'],
            y=position_data_orig[i]['Y'],
            mode='markers',
            marker=dict(
                symbol=marker_styles[i % len(marker_styles)],  # Cycle through marker styles
                color=color_for_lap,
                size=5  # Optional: Adjust marker size
            ),
            name=f'Original Data Lap {lap_number}',
            customdata=position_data_orig[i]['Distance'],
            hoverinfo='skip',
            showlegend=False
        ), row=1, col=2)

    fig.add_trace(go.Scatter(x=x_left - offset_x, y=y_left - offset_y, mode='lines', line=dict(color='black'),name='Right Boundary',showlegend=False, hoverinfo='skip'), row = 1, col = 2)
    fig.add_trace(go.Scatter(x=x_right - offset_x, y=y_right - offset_y, mode='lines',line=dict(color='black'), name='Left Boundary',showlegend=False, hoverinfo='skip'), row = 1, col = 2)
    #fig.add_trace(go.Scatter(x=raceline_data[0] - offset_x, y=raceline_data[1] - offset_y, mode='lines', name='Ideal racing line'), row = 1, col = 2)

    # Arrays to store valid points
    #valid_x_original = []
    #valid_y_original = []

    # Iterate over each point in x_coords_original and y_coords_original
    #for x, y in zip(x_coords_original, y_coords_original):
    #    point = (x, y)

        # Check if the point is inside the right boundary polygon
    #    if left_boundary_path.contains_point(point):
            # Check if the point is outside the left boundary polygon
    #        if not right_boundary_path.contains_point(point):
    #            # If it's inside the right boundary and outside the left boundary, it's valid
    #            valid_x_original.append(x)
    #            valid_y_original.append(y)

    #valid_x = valid_x_original
    #valid_y = valid_y_original

    # Customize layout
    #fig.add_trace(go.Scatter(x=valid_x_original, y=valid_y_original, mode='markers',name='original data',showlegend = False), row = 1, col = 2)
    #fig.add_trace(go.Scatter(x=valid_x, y=valid_y, mode='lines',name=' data f=100',showlegend = False), row = 1, col = 2)
    #fig.add_trace(go.Scatter(x=x_left_shifted, y=y_left_shifted, mode='lines',name='right boundary',showlegend = False), row = 1, col = 2)
    #fig.add_trace(go.Scatter(x=x_right_shifted, y=y_right_shifted, mode='lines',name='left boundary',showlegend = False), row = 1, col = 2)

    ## Update Axes
    fig.update_xaxes(title_text = "X Coordinate (m)", row = 1, col = 2)
    fig.update_yaxes(title_text = "Y Coordinate (m)", row = 1, col = 2)

    return fig

'''
Draws Plot 3, the stacked bar chart.
'''
def draw_stackedbar(fig):
    lap_time_data = sector_times_df[(sector_times_df['Driver'] == DRIVER) & (sector_times_df['LapNumber'] != 65)]
    driver_data = sector_times_long[(sector_times_long['Driver'] == DRIVER) & (sector_times_long['LapNumber'] != 65)]
    for sector, color in zip(['Sector1Time', 'Sector2Time', 'Sector3Time'], ['blue', 'green', 'orange']):
        sector_data = driver_data[driver_data['Sector'] == sector]

        fig.add_trace(
            go.Bar(
                x=sector_data['LapNumber'],
                y=sector_data['Time'],
                name=f"{sector}",
                showlegend = False,
                marker = dict(
                    color = lap_time_data['Compound'].map(lambda x: TIRE_COLOUR.get(x.lower(), 'black')),
                    line=dict(color='black', width=1)
                ),
            ),
            row = 2, col = 1
        )

    ## Update Axes
    fig.update_xaxes(title_text = "Lap Number", row = 2, col = 1)
    fig.update_yaxes(title_text = "Lap Times (s)", range = [0, sector_times_df['LapTime'].max() * 1], row = 2, col = 1)

    return fig

'''
Draws Plot 4, the racelines and boundaries.
'''
def draw_racelines(fig):
    offset_x = 71
    offset_y = 198

    x_left_shifted = x_left - offset_x
    y_left_shifted = y_left - offset_y
    x_right_shifted = x_right - offset_x
    y_right_shifted = y_right - offset_y

    # Create LineString objects for left and right boundaries
    left_boundary = LineString(zip(x_left_shifted, y_left_shifted))
    right_boundary = LineString(zip(x_right_shifted, y_right_shifted))
    distance_along_tangent = 50  # Distance to extend tangents, adjust as needed
    relative_positions_all_laps = []

    for i, lap_number in enumerate(SELECTED_LAPS):
        previous_relative_position = None
        relative_positions = []
        valid_x = position_data_100[i]['X'].values
        valid_y = position_data_100[i]['Y'].values

        # Calculate direction vectors (tangent vectors)
        dx_race = np.diff(valid_x)
        dy_race = np.diff(valid_y)

        # Calculate magnitude of tangent vectors
        norm_race_mag = np.sqrt(dx_race**2 + dy_race**2)

        # Replace zeros with a small value to avoid nan in tangent_x/y_race bc dividing by 0
        norm_race_mag[norm_race_mag == 0] = 1e-10

        # Normalize vectors to get unit vectors
        norm_race_x = dx_race / norm_race_mag
        norm_race_y = dy_race / norm_race_mag

        # Calculate the normal vectors (perpendicular to tangent)
        tangent_x_race = -norm_race_y
        tangent_y_race = norm_race_x

        for j, (x, y) in enumerate(zip(valid_x[:-1], valid_y[:-1])):
            point = Point(x, y)
            #print(j)
            # Access tangent vector components
            current_tangent_x_race = tangent_x_race[j]
            current_tangent_y_race = tangent_y_race[j]

            if(current_tangent_y_race == 0 and current_tangent_x_race == 0):
                relative_positions.append(previous_relative_position)
                continue
            #Make a line that follows the tangent vector
            tangent_line = LineString([Point(x - current_tangent_x_race * distance_along_tangent, y - current_tangent_y_race * distance_along_tangent),
                                    Point(x + current_tangent_x_race * distance_along_tangent, y + current_tangent_y_race * distance_along_tangent)])

            # Find intersection points with boundaries
            intersection_inner = tangent_line.intersection(left_boundary)
            intersection_outer = tangent_line.intersection(right_boundary)

            # Make sure that we take the first point if it is multipoint
            if intersection_inner.geom_type != 'Point':
                if intersection_inner.is_empty:  # no intersection
                    relative_positions.append(previous_relative_position)
                    continue
                else:
                    # Find nearest point in MultiPoint to original point
                    nearest_point = nearest_points(Point(x, y), intersection_inner)[1]
                    intersection_inner = nearest_point  # Assign the nearest point

            if intersection_outer.geom_type != 'Point':
                if intersection_outer.is_empty:  # no intersection
                    relative_positions.append(previous_relative_position)
                    continue
                else:
                    # Find nearest point in MultiPoint to original point
                    nearest_point = nearest_points(Point(x, y), intersection_outer)[1]
                    intersection_outer = nearest_point  # Assign the nearest point


            inner_x, inner_y = intersection_inner.x, intersection_inner.y
            outer_x, outer_y = intersection_outer.x, intersection_outer.y

            # Calculate distances to intersection points
            distance_to_inner = point.distance(intersection_inner)
            distance_to_outer = point.distance(intersection_outer)

            # Calculate relative position
            total_width = distance_to_inner + distance_to_outer
            relative_position = distance_to_outer / total_width
            relative_positions.append(relative_position)
            previous_relative_position = relative_position

        relative_positions_all_laps.append(relative_positions)

    lap_time_data = sector_times_df[(sector_times_df['Driver'] == DRIVER) & (sector_times_df['LapNumber'] != 65)]

    for i, lap_number in enumerate(SELECTED_LAPS):
        # Access distances for the current lap
        distances = position_data_100[i]['Distance'].values
        compound_on_lap = lap_time_data['Compound'].values[lap_number]  # Get the compound name (e.g., 'soft')
        color_for_lap = TIRE_COLOUR.get(compound_on_lap.lower(), 'black')  # Map to color 
        # Access relative_positions for the current lap
        # (Assuming you have a list of relative_positions for each lap)
        # Example: relative_positions_all_laps = [relative_positions_lap1, relative_positions_lap2, ...]
        relative_positions = relative_positions_all_laps[i]
        #Check if relative_positions and distances have the same length
        min_len = min(len(relative_positions), len(distances))
        relative_positions = relative_positions[:min_len]
        distances = distances[:min_len]

        fig.add_trace(go.Scatter(
            x=distances,
            y=relative_positions,
            mode='lines',
            line=dict(color=color_for_lap),
            name=f'Lap {lap_number}',
            hovertemplate=('Rel. Position: %{y:.1f}<br>'  # Display relative position
        )
        ), row = 2, col = 2)

    max_distance = max(distances)  # Get the maximum distance value

    # Add horizontal lines for boundaries (modified)
    fig.add_shape(go.layout.Shape(
        type="line",
        x0=0,
        x1=max_distance,  # Adjust x1 to cover the desired x-axis range
        y0=0,
        y1=0,
        line=dict(color="Black", width=3, dash="solid"),  # Changed color, width, and dash
    ), row = 2, col = 2)
    fig.add_shape(go.layout.Shape(
        type="line",
        x0=0,
        x1=max_distance,  # Adjust x1 to cover the desired x-axis range
        y0=1,
        y1=1,
        line=dict(color="Black", width=3, dash="solid"),  # Changed color, width, and dash
    ), row = 2, col = 2)

    # Add annotations for boundary labels (modified)
    fig.add_annotation(
        x=max_distance * 0.16, # Adjust the position to be inside the plot
        y=0.05,  # Position slightly below the line (adjust as needed)
        text="Outer Boundary",
        showarrow=False,
        font=dict(size=12),
        xanchor="right",  # Anchor to the right to avoid overlapping with the line
        yanchor="top",
        row = 2, col = 2   # Anchor to the top to position it below the line
        )
    fig.add_annotation(
        x=max_distance * 0.16, # Adjust the position to be inside the plot
        y=0.95,  # Position slightly above the line (adjust as needed)
        text="Inner Boundary",
        showarrow=False,
        font=dict(size=12),
        xanchor="right",  # Anchor to the right to avoid overlapping with the line
        yanchor="bottom",# Anchor to the bottom to position it above the line
        row = 2, col = 2)

    ## Update Axes
    fig.update_xaxes(title_text = "Distance (m)", row = 2, col = 2)
    fig.update_yaxes(title_text = "Relative Position On Racetrack", row = 2, col = 2)

    return fig

'''
**Slaps top** This bad boy creates our whole visual.
'''
def create_visual():
    ## Create 4 plots, with appropriate names.
    fig = make_subplots(
        rows = 2, cols = 2,
        subplot_titles=["Lap Times and Events", "Track Map", "Sector Times with Compounds", "Relative Position Between Boundaries"],
        vertical_spacing = 0.1,
       specs=[
            [{'secondary_y': True}, {'secondary_y': True}],
            [{'secondary_y': True}, {'secondary_y': True}]
        ]
    )

    ## Draw Legend
    fig = draw_legend(fig)

    ## Draw Plot 1
    fig = draw_scatterplot(fig)

    ## Draw Plot 2
    fig = draw_racetrack(fig)

    ## Draw Plot 3
    fig = draw_stackedbar(fig)

    ## Draw Plot 4
    fig = draw_racelines(fig)

    ## Update Layout
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray'),
        barmode='stack',
        updatemenus=[dict(
            active=0,
            direction="down",
            x=1.15,
            xanchor="left",
            y=1.10,
            yanchor="top",
        )],
        hovermode="x unified",
        height=1000,
        legend=dict(
            x=1,
            y=1,  
            traceorder='normal',
            orientation='v', 
            xanchor='left',
            yanchor='top',
            bgcolor="White",
            bordercolor="Black",
            borderwidth=2
        ),
    )

    return fig

## data preprocessing (mads + max's codes)
telemetry_100, telemetry_original = get_telemetry_and_positions()
x_right, y_right, x_left, y_left = calculate_boundaries()
sector_times_df, sector_times_long = sector_time_calculation()

## weather processing (rouvin's weather code)

# rainfall_index_change = [0] # CHANGE ME TO THE LAP NUMBERS WHERE IT RAINS
# approximate_weather_change_laps = estimate_weather_changes(rainfall_index_change)
# weather_emojis = ['☀️' if i % 2 == 0 else '🌧️' for i in range(len(approximate_weather_change_laps))] # alternating between sun and rain

## weather processing part 2 (Lukas' new weather code)
# Conditions in data: 'Rain' 'Partially cloudy' 'Clear' 'Overcast'
# Map the weather conditions to emojis
# def get_weather_emoji(condition):
#   if condition == 'Rain':
#        return '🌧️'
#    elif condition == 'Partially cloudy':
#        return '⛅'
#    elif condition == 'Clear':
#        return '☀️'
#    elif condition == 'Overcast':
#        return '☁️'
#    else:
#        return '❓'

# Create a list of emojis based on the 'Conditions' column
# weather_emojis = [get_weather_emoji(cond) for cond in laps['Conditions']]

## lap event processing (lukas's code)
lap_events = process_lap_events()
lap_event_emojis = [EVENT_EMOJIS.get(status, '') for status in lap_events['TrackStatusHierarchy']]

## create the figure itself
fig = create_visual()

'''
Use Dash to make our graphs.
Initialize our fig and a variable to store the selected points.
'''
app = Dash(__name__)
app.layout = html.Div([
    dcc.Graph(id='interactive-plot', figure=fig),
    dcc.Store(id='selected-points', data=[]),
    dcc.Store(id='current-points', data=[]),  # Store for current points
    dcc.Store(id='prev-points', data=[])      # Store for previous points
])

'''
Method to update the highlighting of the graphs.
Current limitation: Each time you select new points, it will be added to the `selected_data` variable. 
You you cannot 'unselect' points without refreshing the page.
'''
@app.callback(
    [Output('interactive-plot', 'figure'),
     Output('selected-points', 'data'),
     Output('current-points', 'data'),
     Output('prev-points', 'data')],
    [Input('interactive-plot', 'selectedData')],
    [State('selected-points', 'data'),
     State('current-points', 'data'),
     State('prev-points', 'data')]
)
def update_graph(selected_data, stored_points, current_points, prev_points):
    isSelected = selected_data is not None and 'points' in selected_data and len(selected_data['points']) != 0
    updated_fig = go.Figure(fig) 
    if isSelected: # if there is data selected
        if len(current_points) == 0 and len(prev_points) == 0: # first selection
            current_points = [p['pointIndex'] for p in selected_data['points']]
            updated_fig.update_traces(
                selectedpoints=current_points,
                marker={"opacity": 1},  
                unselected_marker={"opacity": 0.3}, 
                row = 1, col = 1
            )
            updated_fig.update_traces(
                selectedpoints=current_points,
                marker={"opacity": 1},  
                unselected_marker={"opacity": 0.3}, 
                row = 2, col = 1
            )

        else: # any subsequent selection
            prev_points = current_points
            current_points = [p['pointIndex'] for p in selected_data['points']]
            updated_fig.update_traces(
                selectedpoints=current_points,
                marker={"opacity": 1},  
                unselected_marker={"opacity": 0.3}, 
                row = 1, col = 1
            )
            updated_fig.update_traces(
                selectedpoints=current_points,
                marker={"opacity": 1},  
                unselected_marker={"opacity": 0.3}, 
                row = 2, col = 1
            )
    else: # nothing is selected
        if len(current_points) > 0:  # Retain previous state
            updated_fig.update_traces(
                selectedpoints=current_points,
                marker={"opacity": 1},  
                unselected_marker={"opacity": 0.3}, 
                row = 1, col = 1
            )
            updated_fig.update_traces(
                selectedpoints=current_points,
                marker={"opacity": 1},  
                unselected_marker={"opacity": 0.3}, 
                row = 2, col = 1
            )
        else:  # Reset all points to full opacity if no points are selected
            current_points = []
            prev_points = []
            isSelected = False
            updated_fig.update_traces(
                marker={"opacity": 1},  # Reset all points to full opacity
                selectedpoints=None,
                row = 1, col = 1 # Clear selection highlights
            )
            updated_fig.update_traces(
                marker={"opacity": 1},  # Reset all points to full opacity
                selectedpoints=None,
                row = 2, col = 1 # Clear selection highlights
            )

    return updated_fig, stored_points, current_points, prev_points

if __name__ == '__main__':
    app.run_server(debug=True)