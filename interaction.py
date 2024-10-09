import fastf1 as ff1
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load race constants
YEAR = 2019
GRAND_PRIX = 'German Grand Prix'
SESSION_TYPE = 'R' 

# Load session and data
race = ff1.get_session(YEAR, GRAND_PRIX, SESSION_TYPE)
race.load()
laps = race.laps

# Define parameters that are interesting
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

## Check stuff via print
# driver_lap_counts = sector_times_df['Driver'].value_counts()
# compound_counts = sector_times_df['Compound']
# print(nat_rows[['Driver','LapNumber','Sector1Time', 'Sector2Time', 'Sector3Time', 'LapTime', 'Compound']])
# print(driver_lap_counts)
# print(compound_counts)

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

def create_interactive_plot_with_subplots():
    fig = make_subplots(
        rows = 2, cols = 1,
        subplot_titles = ("Sector Times", "Lap Time"),
        vertical_spacing = 0.15
    )

    drivers = sector_times_df['Driver'].unique()

    for i, driver in enumerate(drivers):
        driver_data = sector_times_long[sector_times_long['Driver'] == driver]
        
        for sector, color in zip(['Sector1Time', 'Sector2Time', 'Sector3Time'], ['blue', 'green', 'orange']):
            sector_data = driver_data[driver_data['Sector'] == sector]
            fig.add_trace(
                go.Scatter(
                    x = sector_data['LapNumber'],
                    y = sector_data['Time'],
                    mode = 'lines+markers',
                    name = f"{driver} - {sector}",
                    visible = (i == 0),
                    line = dict(shape = 'linear', color = color)
                ),
                row = 1, col = 1
            )

        lap_time_data = sector_times_df[sector_times_df['Driver'] == driver]
        lap_compound_info = lap_time_data['Compound']  # Compound info for lap times
        
        fig.add_trace(
            go.Scatter(
                x = lap_time_data['LapNumber'],
                y = lap_time_data['LapTime'],
                mode = 'lines+markers',
                name = f"{driver} - Lap Time",
                visible = (i == 0),
                line = dict(shape = 'linear', color = 'firebrick'),
                text = lap_compound_info,  # Add compound info for hover text
                hovertemplate = 'Lap: %{x}<br>Lap Time: %{y}<br>Compound: %{text}',  # Display compound in hover
            ),
            row = 2, col = 1
        )

    dropdown_buttons = []
    for i, driver in enumerate(drivers):
        visibility = [False] * len(fig.data)
        visibility[i * 4:i * 4 + 4] = [True, True, True, True]

        dropdown_buttons.append(
            dict(
                label = driver,
                method = "update",
                args = [{"visible": visibility}, 
                        {"title": f"Sector Times and Lap Time for {driver}"}], 
            )
        )

    fig.update_layout(
        updatemenus = [
            dict(
                active = 0,
                buttons = dropdown_buttons,
                direction = "down", 
                x = 1.15, 
                xanchor = "left",
                y = 1.15,
                yanchor = "top",
            ),
        ],
        title = f"Sector Times and Lap Time for {drivers[0]}",
        hovermode = "x unified",
        height = 800
    )

    fig.update_xaxes(title_text = "Lap Number", row = 2, col = 1)
    fig.update_yaxes(title_text = "Sector Time (s)", range = [0, sector_times_df[['Sector1Time', 'Sector2Time', 'Sector3Time']].max().max() * 1.1], row = 1, col = 1)
    fig.update_yaxes(title_text = "Lap Time (s)", range = [0, sector_times_df['LapTime'].max() * 1.1], row = 2, col = 1)
    
    fig.show()


create_interactive_plot_with_subplots()