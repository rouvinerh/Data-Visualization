import fastf1 as ff1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from plotly.subplots import make_subplots
import math

### Set constants here for code ###
YEAR = 2019
GRAND_PRIX = 'German Grand Prix'
SESSION_TYPE = 'R' 
DRIVER = 'VER'

### Load session, race data and weather data ###
race = ff1.get_session(YEAR, GRAND_PRIX, SESSION_TYPE)
race.load(weather = True)
laps = race.laps
weather_data = race.weather_data

### Load CSV Data ###
track_data_url = "https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/tracks/Hockenheim.csv"
df = pd.read_csv(track_data_url)

### Load the ideal racing line data ###
raceline_url = 'https://github.com/TUMFTM/racetrack-database/raw/e59595d1f3573b30d1ded6a08984935b957688e0/racelines/Hockenheim.csv'
raceline_data = pd.read_csv(raceline_url, comment='#', header=None)  # Load the raceline data

### Set laps for drivers (Max's) ###
try:
    # Get laps for the driver
    driver_laps = race.laps.pick_driver(DRIVER)

    # Check if there are any laps available for this driver
    if driver_laps.empty:
        print(f"No laps available for driver {DRIVER}")

    # Extract telemetry for the fastest lap of this driver
    selected_laps = driver_laps.pick_laps(1)
    #selected_laps = driver_laps#.pick_fastest() **************************** ROUVIN HERE SELECT THE LAP(S) YOU WANT
    # **** IM JUST GONNA GO WITH THIS FIRST - Rouvin
    # OR selected_laps = driver_laps.pick_laps(range(10,21))
    # OR selected_laps = driver_laps.pick_laps(1)

    # Get telemetry data at the highest available resolution (100 frequency) and the original one
    telemetry = selected_laps.get_telemetry(frequency=100)  # Use 'original' for highest granularity
    telemetry_original= selected_laps.get_telemetry(frequency='original')  # Use 'original' for highest granularity

    # Check if positional telemetry is available (X, Y data)
    if 'X' in telemetry.columns and 'Y' in telemetry.columns:
        # Extract the x and y coordinates
        x_coords = telemetry['X']/10 # Adjust from 1/10m -> 1m scale
        y_coords = telemetry['Y']/10 # Adjust from 1/10m -> 1m scale


    # Check if positional telemetry is available (X, Y data)
    if 'X' in telemetry_original.columns and 'Y' in telemetry_original.columns:
        # Extract the x and y coordinates
        x_coords_original = telemetry_original['X']/10 # Adjust from 1/10m -> 1m scale
        y_coords_original = telemetry_original['Y']/10 # Adjust from 1/10m -> 1m scale

        # Combine them into a DataFrame
        position_data = pd.DataFrame({
            'X': x_coords,
            'Y': y_coords,
            'Time': telemetry['Date']  # Add timestamps to check the data frequency
        })

            # Combine them into a DataFrame
        position_data_original = pd.DataFrame({
            'X': x_coords_original,
            'Y': y_coords_original,
            'Time': telemetry_original['Date']  # Add timestamps to check the data frequency
        })

        # Calculate the time difference between consecutive telemetry points
        time_diff = position_data['Time'].diff().dt.total_seconds()


        # Calculate the time difference between consecutive telemetry points
        time_diff_original = position_data_original['Time'].diff().dt.total_seconds()

        # # Print statistics about the frequency
        # print(f"Telemetry for driver {DRIVER} at 100 frequency:")
        # print(f"Number of data points: {len(position_data)}")
        # print(f"Average time interval between points: {time_diff.mean()} seconds")
        # print(position_data.head())  # Display first few rows of X, Y, and Time data

        #     # Print statistics about the frequency
        # print(f"Telemetry  for driver {DRIVER} at original frequency:")
        # print(f"Number of data points: {len(position_data_original)}")
        # print(f"Average time interval between points: {time_diff_original.mean()} seconds")
        # print(position_data_original.head())  # Display first few rows of X, Y, and Time data
        # # Exit loop after finding first driver with valid data

    else:
        print(f"No positional data (X, Y) available for driver {DRIVER}")

except KeyError as e:
    print(f"Telemetry not available for driver {DRIVER}: {e}")
except Exception as e:
    print(f"Error encountered for driver {DRIVER}: {e}")

### Extract the centerline and track width data (Max's) ###
x = df['# x_m'].values  # Adjust based on the actual column name
y = df['y_m'].values    # Adjust based on the actual column name
track_width_right = df['w_tr_right_m'].values
track_width_left = df['w_tr_left_m'].values

# print(track_width_right[1])

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

# Plot the track
# plt.plot(x, y, label='Centerline', color='black')
# plt.plot(x_left, y_left, label='Left Boundary', color='blue')
# plt.plot(x_right, y_right, label='Right Boundary', color='red')

# plt.title("Hockenheim Track Outline")
# plt.xlabel("X [m]")
# plt.ylabel("Y [m]")
# plt.legend()
# plt.show()

### Define parameters that are interesting (Mad's) ###
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

def create_combined_plot():
    fig = make_subplots(
        rows = 2, cols = 2,
        subplot_titles=["Lap Times and Events", "Track Map", "Sector Times with Compounds"],
        vertical_spacing = 0.1,
       specs=[
            [{'secondary_y': True}, {'secondary_y': True}],
            [{'secondary_y': True}, {'secondary_y': True}]
        ]
    )

    ### Colours of Tire Compounds ###
    compound_to_color = {
        'soft': '#377eb8',        
        'medium': '#ff7f00',      
        'hard': '#4daf4a',        
        'intermediate': '#f781bf',
        'wet': '#a65628'          
    }

    ### Shapes of Tire Compounds for Plot 1 ###
    compound_to_marker = {
        'soft': 'circle',
        'medium': 'square',
        'hard': 'triangle-up',
        'intermediate': 'diamond',
        'wet': 'x'
    }

    ### For Legend on the right ###
    for compound, color in compound_to_color.items():
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

    ### Plot 1: Scatterplot ###
    lap_time_data = sector_times_df[sector_times_df['Driver'] == DRIVER]
    fig.add_trace(go.Scatter(
        x = lap_time_data['LapNumber'],
        y = lap_time_data['LapTime'],
        mode = 'markers',
        name = '<extra></extra>', 
        marker = dict(
            size = 8,
            color = lap_time_data['Compound'].map(lambda x: compound_to_color.get(x.lower(), 'black')),
            symbol = lap_time_data['Compound'].map(lambda  x: compound_to_marker.get(x.lower(), 'cross'))
        ),
        hovertemplate = 'Lap: %{x}<br>Lap Time: %{y:.2f} seconds<br>Tire Compound: %{text}',
        text = lap_time_data['Compound'],
        showlegend = False
    ), row = 1, col = 1)

    ### Plot 1: Scatterplot Conditions ### 
    ### Rain
    fig.add_trace(
        go.Scatter(
            x = approximate_laps,  
            y = [180] * len(approximate_laps), 
            mode = 'text',  
            text = ['🌧️'] * len(approximate_laps),
            textposition = 'middle center',
            name = 'Rainfall Change Points',
            showlegend = False,
            hoverinfo='none',
            textfont=dict(
                size=25 
            )
        ),
        row = 1, col= 1, secondary_y=True
    )

    ### Flags
    # fig.add_trace(
    #     go.Scatter(
    #         x = approximate_laps,  
    #         y = [180] * len(approximate_laps), 
    #         mode = 'text',  
    #         text = ['🚩'] * len(approximate_laps),
    #         textposition = 'middle center',
    #         name = 'Rainfall Change Points',
    #         showlegend = False,
    #         hoverinfo='none',
    #     ),
    #     row = 1, col= 1, secondary_y=True
    # )

    ### Plot 2: Race Map ###
    offset_x = 71
    offset_y = 198

    # Add the telemetry data as a line plot
    fig.add_trace(go.Scatter(x=x_coords, y=y_coords, mode='lines',name=' data f=100',showlegend = False), row = 1, col = 2)
    fig.add_trace(go.Scatter(x=x_coords_original, y=y_coords_original, mode='markers',name='original data',showlegend = False), row = 1, col = 2)
    fig.add_trace(go.Scatter(x=x_left-offset_x, y=y_left-offset_y, mode='lines',name='right boundary',showlegend = False), row = 1, col = 2)
    fig.add_trace(go.Scatter(x=x_right-offset_x, y=y_right-offset_y, mode='lines',name='left boundary',showlegend = False), row = 1, col = 2)
    fig.add_trace(go.Scatter(x=raceline_data[0]-offset_x, y=raceline_data[1]-offset_y, mode='lines',name='Ideal racing line',showlegend = False, ), row = 1, col = 2)

    x_left_shifted = x_left - offset_x
    y_left_shifted = y_left - offset_y
    x_right_shifted = x_right - offset_x
    y_right_shifted = y_right - offset_y

    # Combine x and y coordinates into polygon points for left and right boundaries
    right_boundary_points = np.column_stack((x_right_shifted, y_right_shifted))
    left_boundary_points = np.column_stack((x_left_shifted, y_left_shifted))

    # Combine x and y coordinates into polygon points for left and right boundaries
    right_boundary_points = np.column_stack((x_right_shifted, y_right_shifted))
    left_boundary_points = np.column_stack((x_left_shifted, y_left_shifted))

    # Create Path objects representing the right and left boundary polygons
    right_boundary_path = Path(right_boundary_points)
    left_boundary_path = Path(left_boundary_points)

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

    # Arrays to store valid points
    valid_x = []
    valid_y = []

    # Iterate over each point in x_coords_original and y_coords_original
    for x, y in zip(x_coords_original, y_coords_original):
        point = (x, y)

        # Check if the point is inside the right boundary polygon
        if left_boundary_path.contains_point(point):
            # Check if the point is outside the left boundary polygon
            if not right_boundary_path.contains_point(point):
                # If it's inside the right boundary and outside the left boundary, it's valid
                valid_x.append(x)
                valid_y.append(y)

    # Customize layout
    fig.add_trace(go.Scatter(x=valid_x_original, y=valid_y_original, mode='markers',name='original data',showlegend = False), row = 1, col = 2)
    fig.add_trace(go.Scatter(x=valid_x, y=valid_y, mode='lines',name=' data f=100',showlegend = False), row = 1, col = 2)
    fig.add_trace(go.Scatter(x=x_left_shifted, y=y_left_shifted, mode='lines',name='right boundary',showlegend = False), row = 1, col = 2)
    fig.add_trace(go.Scatter(x=x_right_shifted, y=y_right_shifted, mode='lines',name='left boundary',showlegend = False), row = 1, col = 2)

    ### Plot 3: Stacked Bar Chart ###
    driver_data = sector_times_long[sector_times_long['Driver'] == DRIVER]
    for sector, color in zip(['Sector1Time', 'Sector2Time', 'Sector3Time'], ['blue', 'green', 'orange']):
        sector_data = driver_data[driver_data['Sector'] == sector]

        fig.add_trace(
            go.Bar(
                x=sector_data['LapNumber'],
                y=sector_data['Time'],
                name=f"{sector}",
                showlegend = False,
                marker = dict(
                    color = lap_time_data['Compound'].map(lambda x: compound_to_color.get(x.lower(), 'black')),
                    line=dict(color='black', width=1)
                ),
            ),
            row = 2, col = 1
        )
    
    ### Plot 4: Line Graph ###
    # To be implemented #

    ### Update Layout of the Graphs ###
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

    ### Update Axis ###
    fig.update_xaxes(title_text = "Lap Number", row = 1, col = 1)
    fig.update_xaxes(title_text = "X-Coordinate", row = 1, col = 2)
    fig.update_yaxes(title_text = "Y Coordinate", row = 1, col = 2)
    fig.update_yaxes(title_text = "Lap Times", range = [0, sector_times_df['LapTime'].max() * 1.1], row = 1, col = 1)
    fig.update_xaxes(range = [-1, sector_times_df['LapNumber'].max()], row = 1, col = 1)
    fig.show()

create_combined_plot()