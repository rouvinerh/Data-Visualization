import math
import fastf1 as ff1
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.path import Path
from plotly.subplots import make_subplots
from shapely.geometry import Point, Polygon, LineString
from dash import Dash, dcc, html, Input, Output, callback

'''
Constants that define how the code runs.
Change this if you'd like to see a different course, driver or lap.
'''
YEAR = 2019
GRAND_PRIX = 'German Grand Prix'
SESSION_TYPE = 'R' 
DRIVER = 'HAM'
SELECTED_LAPS = [29, 55, 32]

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
    'soft': '#377eb8',        
    'medium': '#ff7f00',      
    'hard': '#4daf4a',        
    'intermediate': '#f781bf',
    'wet': '#a65628'          
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
laps = race.laps
weather_data = race.weather_data
driver_laps = race.laps.pick_driver(DRIVER)
selected_laps = driver_laps.pick_laps(SELECTED_LAPS)
position_data_100 = []  # List to store position data for each lap at 100Hz
position_data_orig = [] # List to store position data for each lap at original frequency
x_coords_original = []
y_coords_original = []

'''
Load CSVs about the Hockenheim track.
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
def sectortime_calculation():
    sector_times_df = laps[['Driver', 'LapNumber', 'Sector1Time', 'Sector2Time', 'Sector3Time', 'LapTime', 'Compound']].copy()

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
        sector_times_df[['Sector1Time', 'Sector2Time', 'Sector3Time', 'LapTime', 'Compound']].isna().any(axis = 1)
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
def estimate_weather_changes():
    ### Calculate total number of minutes race takes ###
    max_laps_driver = laps.groupby('Driver')['LapNumber'].max().idxmax()
    max_laps_driver_laps = laps[laps['Driver'] == max_laps_driver]
    total_race_time_seconds = max_laps_driver_laps['LapTime'].fillna(pd.Timedelta(0)).sum().total_seconds()
    total_race_time_minutes = math.ceil(total_race_time_seconds / 60)
    max_laps_driver_laps = max_laps_driver_laps.copy()  
    max_laps_driver_laps['CumulativeLapTime'] = max_laps_driver_laps['LapTime'].cumsum().dt.total_seconds()

    ### Track minutes where changes in Rainfall happens ###
    change_indexes = [0]
    for i in range(1, total_race_time_minutes):
        if weather_data['Rainfall'][i] != weather_data['Rainfall'][i - 1]:
            change_indexes.append(i)

    ### Approximate the minute to the closest lap where Rainfall changed ###
    approximate_laps = []
    for minute in change_indexes:
        target_time_seconds = minute * 60 
        lap_data = max_laps_driver_laps[max_laps_driver_laps['CumulativeLapTime'] >= target_time_seconds].iloc[0]
        lap_number = lap_data['LapNumber']
        if lap_number > 1:
            previous_lap_time = max_laps_driver_laps[max_laps_driver_laps['LapNumber'] == lap_number - 1]['CumulativeLapTime'].values[0]
        else:
            previous_lap_time = 0 
        
        lap_fraction = (target_time_seconds - previous_lap_time) / (lap_data['CumulativeLapTime'] - previous_lap_time)
        exact_lap = lap_number - 1 + lap_fraction
        
        approximate_laps.append(round(exact_lap, 1))

    return approximate_laps

'''
Process events that happened per lap in the race.
Classifies them accordingly based on EVENT_HIERARCHY.
Returns the lap_events
'''
def process_lap_events():
    lap_events = laps[(laps['Driver'] == 'VER')]
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
    fig.add_trace(go.Scatter(
        x = lap_time_data['LapNumber'],
        y = lap_time_data['LapTime'],
        mode = 'markers',
        name = '<extra></extra>', 
        marker = dict(
            size = 8,
            color = lap_time_data['Compound'].map(lambda x: TIRE_COLOUR.get(x.lower(), 'black')),
            symbol = lap_time_data['Compound'].map(lambda  x: TIRE_SHAPE.get(x.lower(), 'cross'))
        ),
        hovertemplate = 'Lap: %{x}<br>Lap Time: %{y:.2f} seconds<br>Tire Compound: %{text}',
        text = lap_time_data['Compound'],
        showlegend = False
    ), row = 1, col = 1)

    ### Plot 1: Scatterplot Conditions ### 
    ## Rain
    fig.add_trace(
        go.Scatter(
            x = approximate_weather_change_laps,  
            y = [180] * len(approximate_weather_change_laps), 
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
            y = [160] * len(lap_events), 
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

    ### Update axes
    fig.update_xaxes(range=[-1, 70], title_text = "Lap Number", row = 1, col = 1)
    fig.update_yaxes(title_text = "Lap Times (s)", range = [0, sector_times_df['LapTime'].max() * 1.1], row = 1, col = 1)

    return fig

'''
Draws Plot 2, the racetrack.
'''
def draw_racetrack(fig):
    offset_x = 71
    offset_y = 198

    x_left_shifted = x_left - offset_x
    y_left_shifted = y_left - offset_y
    x_right_shifted = x_right - offset_x
    y_right_shifted = y_right - offset_y

    # Combine x and y coordinates into polygon points for left and right boundaries
    right_boundary_points = np.column_stack((x_right_shifted, y_right_shifted))
    left_boundary_points = np.column_stack((x_left_shifted, y_left_shifted))

    # Create Path objects representing the right and left boundary polygons
    right_boundary_path = Path(right_boundary_points)
    left_boundary_path = Path(left_boundary_points)

    inner_boundary = Polygon(zip(x_left_shifted, y_left_shifted))
    outer_boundary = Polygon(zip(x_right_shifted, y_right_shifted))

    for i in range(len(position_data_100)):
        # Create a mask for valid points (Check if points are within outer but not inner boundary)
        mask_100 = position_data_100[i].apply(lambda row: inner_boundary.contains(Point(row['X'], row['Y'])) and not outer_boundary.contains(Point(row['X'], row['Y'])), axis=1)
        mask_orig = position_data_orig[i].apply(lambda row: inner_boundary.contains(Point(row['X'], row['Y'])) and not outer_boundary.contains(Point(row['X'], row['Y'])), axis=1)

        # Update the DataFrames with filtered data
        position_data_100[i] = position_data_100[i][mask_100]
        position_data_orig[i] = position_data_orig[i][mask_orig]

    for i, lap_number in enumerate(SELECTED_LAPS):
        fig.add_trace(go.Scatter(
            x=position_data_100[i]['X'],
            y=position_data_100[i]['Y'],
            mode='lines',
            name=f'Data f=100 Lap {lap_number}'
        ), row = 1, col = 2)
        fig.add_trace(go.Scatter(
            x=position_data_orig[i]['X'],
            y=position_data_orig[i]['Y'],
            mode='markers',
            name=f'Original Data Lap {lap_number}'
        ), row = 1, col = 2)

    fig.add_trace(go.Scatter(x=x_left - offset_x, y=y_left - offset_y, mode='lines', name='Right Boundary'), row = 1, col = 2)
    fig.add_trace(go.Scatter(x=x_right - offset_x, y=y_right - offset_y, mode='lines', name='Left Boundary'), row = 1, col = 2)
    fig.add_trace(go.Scatter(x=raceline_data[0] - offset_x, y=raceline_data[1] - offset_y, mode='lines', name='Ideal racing line'), row = 1, col = 2)

    # Arrays to store valid points
    valid_x_original = []
    valid_y_original = []

    # Iterate over each point in x_coords_original and y_coords_original
    for x, y in zip(x_coords_original, y_coords_original):
        point = (x, y)

        # Check if the point is inside the right boundary polygon
        if left_boundary_path.contains_point(point):
            # Check if the point is outside the left boundary polygon
            if not right_boundary_path.contains_point(point):
                # If it's inside the right boundary and outside the left boundary, it's valid
                valid_x_original.append(x)
                valid_y_original.append(y)

    valid_x = valid_x_original
    valid_y = valid_y_original

    # Customize layout
    fig.add_trace(go.Scatter(x=valid_x_original, y=valid_y_original, mode='markers',name='original data',showlegend = False), row = 1, col = 2)
    fig.add_trace(go.Scatter(x=valid_x, y=valid_y, mode='lines',name=' data f=100',showlegend = False), row = 1, col = 2)
    fig.add_trace(go.Scatter(x=x_left_shifted, y=y_left_shifted, mode='lines',name='right boundary',showlegend = False), row = 1, col = 2)
    fig.add_trace(go.Scatter(x=x_right_shifted, y=y_right_shifted, mode='lines',name='left boundary',showlegend = False), row = 1, col = 2)

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

    for i, lap_number in enumerate(SELECTED_LAPS):
    # Get valid x and y coordinates for the current lap
        valid_x = position_data_100[i]['X'].values
        valid_y = position_data_100[i]['Y'].values

        # Calculate direction vectors (tangent vectors)
        dx_race = np.diff(valid_x)
        dy_race = np.diff(valid_y)

        # Calculate magnitude of tangent vectors
        tangent_norm_race = np.sqrt(dx_race**2 + dy_race**2)

        # Normalize tangent vectors to get unit vectors
        tangent_x_race = dx_race / tangent_norm_race
        tangent_y_race = dy_race / tangent_norm_race

        # Calculate normal vectors (perpendicular to tangent)
        normal_x_race = -tangent_y_race
        normal_y_race = tangent_x_race

        # Calculate angle changes
        dx_race = np.diff(normal_x_race)
        dy_race = np.diff(normal_y_race)
        change = np.sqrt(dx_race**2 + dy_race**2)

        # Get distance values for the current lap
        distances = position_data_100[i]['Distance'].values

    left_boundary = LineString(zip(x_left_shifted, y_left_shifted))
    right_boundary = LineString(zip(x_right_shifted, y_right_shifted))

    relative_positions_all_laps = []

    for i, lap_number in enumerate(SELECTED_LAPS):

        relative_positions = []
        for x, y in zip(position_data_100[i]['X'].values, position_data_100[i]['Y'].values):
            point = Point(x, y)

            # Calculate distances to boundaries
            distance_to_inner = point.distance(left_boundary)
            distance_to_outer = point.distance(right_boundary)

            # Calculate relative position
            total_width = distance_to_inner + distance_to_outer
            relative_position = distance_to_inner / total_width

            relative_positions.append(relative_position)

        relative_positions_all_laps.append(relative_positions)
    for i, lap_number in enumerate(SELECTED_LAPS):
        # Access distances for the current lap
        distances = position_data_100[i]['Distance'].values

        # Access relative_positions for the current lap
        # (Assuming you have a list of relative_positions for each lap)
        # Example: relative_positions_all_laps = [relative_positions_lap1, relative_positions_lap2, ...]
        relative_positions = relative_positions_all_laps[i]

        fig.add_trace(go.Scatter(
            x=distances,
            y=relative_positions,
            mode='lines',
            name=f'Lap {lap_number}'
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
        x=max_distance * 0.15, # Adjust the position to be inside the plot
        y=0.05,  # Position slightly below the line (adjust as needed)
        text="Right Boundary",
        showarrow=False,
        font=dict(size=12),
        xanchor="right",  # Anchor to the right to avoid overlapping with the line
        yanchor="top", row = 2, col = 2   # Anchor to the top to position it below the line
    )
    fig.add_annotation(
        x=max_distance * 0.13, # Adjust the position to be inside the plot
        y=0.95,  # Position slightly above the line (adjust as needed)
        text="Left Boundary",
        showarrow=False,
        font=dict(size=12),
        xanchor="right",  # Anchor to the right to avoid overlapping with the line
        yanchor="bottom", row = 2, col = 2  # Anchor to the bottom to position it above the line
    )
    fig.update_xaxes(title_text = "Distance (m)", row = 2, col = 2)
    fig.update_yaxes(title_text = "Relative Position (0 = Outer Boundary, 1 = Inner Boundary)", row = 2, col = 2)

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

telemetry_100, telemetry_original = get_telemetry_and_positions()
x_right, y_right, x_left, y_left = calculate_boundaries()
sector_times_df, sector_times_long = sectortime_calculation()
approximate_weather_change_laps = estimate_weather_changes()
lap_events = process_lap_events()
lap_event_emojis = [EVENT_EMOJIS.get(status, '') for status in lap_events['TrackStatusHierarchy']]
weather_emojis = ['☀️' if i % 2 == 0 else '🌧️' for i in range(len(approximate_weather_change_laps))] # alternating between sun and rain.
fig = create_visual()

app = Dash(__name__)
app.layout = html.Div([
    dcc.Graph(id='interactive-plot', figure=fig),
    html.Div(id='output')
])

@app.callback(
    Output('interactive-plot', 'figure'),
    Input('interactive-plot', 'selectedData')
)
def update_graph(selected_data):
    # If no points are selected, reset the opacity for all points
    if not selected_data:
        fig.update_traces(
            row=2, col=1,
            marker=dict(opacity=1)  # Reset to full opacity
        )
        return fig
    
    # Extract selected points data
    selected_points = selected_data['points']
    
    # Extract the x-values of the selected points
    x_selected = [point['x'] for point in selected_points]
    
    # Update the opacity of the traces in row=2, col=1 based on whether the x-value is selected
    fig.update_traces(
        row=2, col=1,
        selectedpoints=x_selected,  # Only these x-values will be selected
        unselected={
            "marker": {"opacity": 0.3}  # Make unselected points transparent
        }
    )
    
    return fig


app.run()


# change intervals for events to boxes at varying y-levels
# highlighting issues