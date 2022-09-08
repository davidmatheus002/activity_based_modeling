"""
Explore what the various ODiN filters do w.r.t. number of trips and purpose distribution.

Note that this is straight ODiN data, so no remapping is done on the parameters.

Created on Thu Oct 14 14:05:54 2021

@author: asharib
"""

import config as cfg
import pandas as pd
import sql_queries as q

# %% Create a dictionary of SQL queries with different filters
# Dictionary of different queries
queries = {}

# Get the main query and break into lines
odin_query = q.odin.split("\n")
# Get rid of empty lines and comments
odin_query = [x for x in odin_query if x != "" and "--" not in x]
queries["main"] = odin_query

# Remove the doel IN (4, 6) filter
all_goals = [x for x in odin_query if "o.doel in (4, 6)" not in x]
queries["all_goals"] = all_goals

# Remove the departure/arrival time filters
all_times = [
    x for x in all_goals if "o.vertuur < 5" not in x and "o.aankuur < 5" not in x
]
queries["all_times"] = all_times

# Remove the weekday filter
all_days = [x for x in all_times if "OR (o.weekdag IN (1, 7))" not in x]
queries["all_days"] = all_days

# %% Get data
data = {}
for k, v in queries.items():
    print(f"Running query: {k}")
    data[k] = pd.read_sql("\n".join(v), cfg.odin_engine)
# %% Determine metrics of interest
# Number of people
opids = []
trips = []
work = []
edu = []
for k, v in data.items():
    print(k)
    opids.append(v.opid.nunique())
    trips.append(v.verplid.nunique())
    work.append(v.loc[v.doel == 2, "verplid"].nunique())
    edu.append(v.loc[v.doel == 7, "verplid"].nunique())
# Create a DataFrame (swap rows/columns, I want rows per query)
df = pd.DataFrame([opids, trips, work, edu]).transpose()
# Name column and indices
df.columns = ["person", "trips", "work_trips", "edu_trips"]
df.index = list(data.keys())

df["ratio_work_edu"] = df.work_trips / df.edu_trips
print(df)
