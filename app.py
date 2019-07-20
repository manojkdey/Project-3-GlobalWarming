from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
app = Flask(__name__)
import pandas as pd 
from sqlalchemy.ext.automap import automap_base
# Import relevant libraries
import calendar
from datetime import datetime
import numpy as np

# Import Matplotlib
import matplotlib.pyplot as plt
import sqlite3



#################################################
# Database Setup
#################################################



conn = sqlite3.connect("db/GlobalWarming")
raw_t = pd.read_sql_query("select Year, 100 * Dec as Dec from GlobalMnTemp;", conn)

ax = plt.gca()
raw_t.plot(kind='line',x='Year',y='Dec', color='Blue', ax=ax)
plt.savefig('images/GlobalMeanTemp.png')
#plt.show()


# Import the Pandas library
import pandas as pd 
# Read in the raw temperature and emissions datasets (they are in CSV format) 
rawt = pd.read_csv('resources/GLB.Ts+dSST.csv', skiprows=0)
raw_e = pd.read_csv('resources/API_EN.ATM.CO2E.PC_DS2_en_csv_v2_41048.csv', skiprows=0)
#raw_e.head()

#From our DataFrame, we will use only the row representing the CO₂
#emissions for the entire world. Like before, 
#we will create a new DataFrame that uses a DateTime index — 
#and then use the raw data to populate it
# Define function to pull value from raw data, using DateIndex from new DataFrame row

def populate_df(row):
    index = str(row['date'].year)
    value = raw_e_world.loc[index]
    return value
  
# Select just the co2 emissions for the 'world', and the columns for the years 1960-2018 
raw_e_world = raw_e[raw_e['Country Name']=='World'].loc[:,'1960':'2018']

# 'Traspose' the resulting slice, making the columns become rows and vice versa
raw_e_world = raw_e_world.T
raw_e_world.columns = ['value']

# Create a new DataFrame with a daterange the same the range for.. 
# the Temperature data (after resampling to years)
date_rng = pd.date_range(start='31/12/1960', end='31/12/2018', freq='y')
e = pd.DataFrame(date_rng, columns=['date'])

# Populate the new DataFrame using the values from the raw data slice
v = e.apply(lambda row: populate_df(row), axis=1)
e['Global CO2 Emissions per Capita'] = v
e.set_index('date', inplace=True)
#e.head()


#DateTime indexes make for convenient slicing of data, let’s select all of our data after the year 2001:
#global_co2=
#e[e.index.year>2011]

#There seems to be a few NaN’s towards the end of our data — lets use Panda’s fillna method to deal with this
e.fillna(method='ffill', inplace=True)
#e[e.index.year>2011]

#e['1984-01-04':'1990-01-06']

# Create figures and axes
fig, ax = plt.subplots(figsize=(10,8))

# Plot co2 emissions data with specific colour and line thickness
ax.plot(e, color='#3393FF', linewidth=2.5)

# Set axis labels and graph title
ax.set(xlabel='Time (years)', ylabel='Emissions (Metric Tons per Capita)',
       title='Global CO2 Emission over Time')

# Enable grid
ax.grid()
plt.savefig('images/Global_CO2_Emission_Since_1960.png')


#The Plotly Python package is an open-source library built on plotly.js — which is in turn built on d3.js.
#In this tutorial, we will be using a wrapper called cufflinks — this makes it easy to use Plotly with Pandas DataFrames.
# Standard plotly imports
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
# Using plotly + cufflinks in offline mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)
rawt = rawt.iloc[:,:13]
rawt.head()

rawt = rawt.iloc[:,:13]
rawt.head()


# Create new dataframe with an index for each month
# First create the date range
date_rng = pd.date_range(start='1/1/1880', end='1/03/2019', freq='M')

type(date_rng[0])

### returns 
### pandas._libs.tslibs.timestamps.Timestamp

# Next create the empty DataFrame, which we will populate using the actual data
t = pd.DataFrame(date_rng, columns=['date'])

# Create a column for the anomoly values
t['Avg_Anomaly_deg_C'] = None


# Set the index of the DataFrame to the date column (DateTime index)
t.set_index('date', inplace=True)

# Show the first few elements
t.head()


# Import relevant libraries
import calendar
from datetime import datetime

# Function definition
def populate_df_with_anomolies_from_row(row):
    year = row['Year']
    #year = row[0]
    # Anomaly values (they seem to be a mixture of strings and floats)
    monthly_anomolies = row.iloc[1:]
    # Abbreviated month names (index names)
    months = monthly_anomolies.index
    for month in monthly_anomolies.index:
        # Get the last day for each month 
        last_day = calendar.monthrange(year,datetime.strptime(month, '%b').month)[1]
        # construct the index with which we can reference our new DataFrame (to populate) 
        date_index = datetime.strptime(f'{year} {month} {last_day}', '%Y %b %d')
        # Populate / set value @ above index, to anomaly value
        t.loc[date_index] = monthly_anomolies[month]
        # Apply function to each row of raw data 
_ = rawt.apply(lambda row: populate_df_with_anomolies_from_row(row), axis=1)

# Show the first few elements of our newly populated DataFrame

#-You may have noticed that the anomaly values seem to be a bit messy, 
#they are a mixture of strings and floats — 
#with a few unusable ‘***’ values mixed in (2019). Let's clean them up
# Import Numpy, a library meant for large arrays - we will use it for its NaN representation 
import numpy as np

# Define function to convert values to floats, and return a 'NaN = Not a Number' if this is not possible
def clean_anomaly_value(raw_value):
    try:
        return float(raw_value)
    except:
        return np.NaN
    
# Apply above function to all anomaly values in DataFrame
t['Avg_Anomaly_deg_C'] = t['Avg_Anomaly_deg_C'].apply(lambda raw_value: clean_anomaly_value(raw_value))

# 'Forward fill' to take care of NaN values
t.fillna(method='ffill', inplace=True)

# Show the first few elements of our newly cleaned DataFrame

t.dtypes

#-Let’s downsample our temperature data into years, the string ‘A’ represents ‘calendar year-end’. 
t.resample('A').mean().head()

# Plot the data - quick and easy - using matplotlib, we will draw prettier graphs later

# Import Matplotlib
import matplotlib.pyplot as plt

# Allow for rendering within notebook

# Create figure, title and plot data
plt.figure(figsize=(10,8))
plt.xlabel('Time')
plt.ylabel('Temperature Anomaly (°Celsius)')
plt.title("Global surface temperature anamolies since 1880")
plt.plot(t, color='#1C7C54', linewidth=1.0)
plt.savefig('images/Global_Monthly_surface_temperature_anamolies_since_1880.png')
plt.show()

#-Let’s downsample our temperature data into years, the string ‘A’ represents ‘calendar year-end’. 
t.resample('A').mean().head()
# Now lets visualize our resampled DataFrame

# Create figure, title and plot resampled data
plt.figure(figsize=(10,8))
plt.xlabel('Time')
plt.ylabel('Temperature Anomaly (°Celsius)')
plt.title("Global_annual_surface_temperature_anamolies_since_1880")
plt.savefig('images/Global_annual_surface_temperature_anamolies_since_1880.png')
plt.plot(t.resample('A').mean(), color='#1C7C54', linewidth=1.0)
plt.show()

#From our DataFrame, we will use only the row representing the CO₂
#emissions for the entire world. Like before, 
#we will create a new DataFrame that uses a DateTime index — 
#and then use the raw data to populate it
# Define function to pull value from raw data, using DateIndex from new DataFrame row

def populate_df(row):
    index = str(row['date'].year)
    value = raw_e_world.loc[index]
    return value
  
# Select just the co2 emissions for the 'world', and the columns for the years 1960-2018 
raw_e_world = raw_e[raw_e['Country Name']=='World'].loc[:,'1960':'2018']
#raw_e_world = raw_e[raw_e['Country Name']=='Iran'].loc[:,'1960':'2018']

# 'Traspose' the resulting slice, making the columns become rows and vice versa
raw_e_world = raw_e_world.T
raw_e_world.columns = ['value']

# Create a new DataFrame with a daterange the same the range for.. 
# the Temperature data (after resampling to years)
date_rng = pd.date_range(start='31/12/1960', end='31/12/2018', freq='y')
e = pd.DataFrame(date_rng, columns=['date'])

# Populate the new DataFrame using the values from the raw data slice
v = e.apply(lambda row: populate_df(row), axis=1)
e['Global CO2 Emissions per Capita'] = v
e.set_index('date', inplace=True)
#DateTime indexes make for convenient slicing of data, let’s select all of our data after the year 2001:
#global_co2=
e[e.index.year>2011]


#There seems to be a few NaN’s towards the end of our data — lets use Panda’s fillna method to deal with this
e.fillna(method='ffill', inplace=True)
e[e.index.year>2011]

    
e['1984-01-04':'1990-01-06']

#Let's plot our temperature data again using Matplotlib,  — adding axis labels and titles, etc.
# import Matplotlib
import matplotlib.pyplot as plt
# Allow for graphs to be displayed in Jupyter notebook

# Resample or temperature data to years (end-of-year)
t_resampled = t.resample('A').mean()

# Create figures and axes
fig, ax = plt.subplots(figsize=(10,8))

# Plot temperature data with specific colour and line thickness
ax.plot(t_resampled, color='#1C7C54', linewidth=2.5)

# Set axis labels and graph title
ax.set(xlabel='Time (years)', ylabel='Temperature Anomaly (deg. Celsius)',
       title='Global Temperature Anomalies')

# Enable grid
plt.savefig('images/Global_annual_temperature_anamolies_since_1880.png')
ax.grid()

# Create figures and axes
fig, ax = plt.subplots(figsize=(10,8))

# Plot co2 emissions data with specific colour and line thickness
ax.plot(e, color='#3393FF', linewidth=2.5)

# Set axis labels and graph title
ax.set(xlabel='Time (years)', ylabel='Emissions (Metric Tons per Capita)',
       title='Global CO2 Emission over Time')
plt.savefig('images/Per_Capita_CO2_Emmission_since1960.png')
# Enable grid
ax.grid()

#The Plotly Python package is an open-source library built on plotly.js — which is in turn built on d3.js.
#In this tutorial, we will be using a wrapper called cufflinks — this makes it easy to use Plotly with Pandas DataFrames.
# Standard plotly imports
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
# Using plotly + cufflinks in offline mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)


#et’s plot both datasets again, this time using Plotly and Cufflinks:
#t.resample('A').mean().iplot(kind='line', xTitle='Time (years)', color='#1C7C54',
#                  yTitle='Temperature Anomaly (deg. Celsius)', title='Global Temperature Anomalies')
#plotly.offline.init_notebook_mode()
#plt.savefig('images/Global_Temperature_ANamoly_Plotly.png')
#ig.write_image("images/Global_Temperature_ANamoly_Plotly.svg")
#plotly.offline.plot(fig, filename='Per_Capita_CO2_Emmission.html')


#Plotting temperature data using Plotly
#e.iplot(kind='line', xTitle='Time (years)', color='#3393FF',
#                  yTitle='Emissions (Metric Tons per Capita)', title='Global CO2 Emission over Time')


#- Template example 
# Render template ha to be implemented 
@app.route('/')
def index():
    return render_template("index.html")
    # Use Pandas to perform the sql query
    
if __name__ == "__main__":
    app.run()