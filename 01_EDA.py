from IPython.display import display

"""
Section 1 — Load Dataset
"""

# In[]:
import pandas as pd
import numpy as np

df = pd.read_csv("data/dataset.csv")

print(df.shape)
df.head()

"""
Section 2 — Dataset Overview
"""

# In[]:
print(df.info())

display(df.sample(5))

"""
Section 3 — Missing Value Analysis
"""

# In[]:
nulls = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percent": round(df.isna().mean()*100,2)
})

nulls.sort_values("missing_percent", ascending=False)

# In[]:
important_cols = [
    "action_taken_timestamp",
    "closed_datetime",
    "junction_name",
    "vehicle_type",
    "violation_type",
    "validation_status"
]

df[important_cols].isna().mean()*100

# In[]:
print("Duplicate rows:", df.duplicated().sum())

# In[]:
df["validation_status"].value_counts(dropna=False)

"""
Section 4 — GPS Validation
"""

# In[]:
df["latitude"].describe()

# In[]:
df["longitude"].describe()

# In[]:
print(df["latitude"].min(), df["latitude"].max())
print(df["longitude"].min(), df["longitude"].max())

"""
Section 5 — Timestamp Conversion
"""

# In[]:
date_cols = [
    "created_datetime",
    "closed_datetime",
    "modified_datetime",
    "action_taken_timestamp",
    "validation_timestamp"
]

for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce")

"""
Section 6 — Date Coverage
"""

# In[]:
print(df["created_datetime"].min())
print(df["created_datetime"].max())

"""
Section 7 — Feature Engineering
"""

# In[]:
df["hour"] = df["created_datetime"].dt.hour
df["day_of_week"] = df["created_datetime"].dt.day_name()
df["month"] = df["created_datetime"].dt.month
df["week"] = df["created_datetime"].dt.isocalendar().week
df["is_weekend"] = (
    df["created_datetime"].dt.dayofweek >= 5
).astype(int)

"""
Section 8 — Violation Types
"""

# In[]:
df["violation_type"].value_counts().head(20)

# In[]:
df["violation_type"].sample(20)

"""
Section 9 — Vehicle Types
"""

# In[]:
df["vehicle_type"].value_counts(dropna=False)

"""
Section 10 — Police Stations
"""

# In[]:
df["police_station"].value_counts().head(20)

"""
Section 11 — Junction Analysis
"""

# In[]:
df["junction_name"].value_counts().head(20)

"""
Section 12 — Hourly Violation Pattern
"""

# In[]:
hourly = df.groupby("hour").size()
hourly.sort_values(ascending=False).head(10)

"""
Section 13 — Day-of-Week Pattern
"""

# In[]:
df.groupby("day_of_week").size().sort_values(ascending=False)

"""
Section 14 — Monthly Trend
"""

# In[]:
df.groupby("month").size()

"""
Section 15 — Response Time Feasibility
"""

# In[]:
response_possible = (
    df["action_taken_timestamp"].notna()
).mean()*100

print(response_possible)

# In[]:
date_cols = [
    "created_datetime",
    "closed_datetime",
    "modified_datetime",
    "action_taken_timestamp",
    "validation_timestamp"
]

for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
        df[col] = df[col].dt.tz_localize(None)

# In[]:
print("Action Timestamp Null %")
print(
    round(
        df["action_taken_timestamp"].isna().mean()*100,
        2
    )
)

print("Closed Datetime Null %")
print(
    round(
        df["closed_datetime"].isna().mean()*100,
        2
    )
)

# In[]:
df["response_time_minutes"] = (
    df["action_taken_timestamp"]
    - df["created_datetime"]
).dt.total_seconds()/60

# In[]:
df["response_time_minutes"].describe()

"""
Section 18 — Closure Time Feasibility
"""

# In[]:
df["closure_time_hours"] = (
    df["closed_datetime"]
    - df["created_datetime"]
).dt.total_seconds()/3600

# In[]:
df["closure_time_hours"].describe()

"""
Section 19 — Spatial Density Readiness
"""

# In[]:
print(
    df[["latitude","longitude"]]
    .dropna()
    .shape
)

"""
Section 20 — Hotspot Candidate Count
"""

# In[]:
df["lat_grid"] = df["latitude"].round(3)
df["lon_grid"] = df["longitude"].round(3)

hotspots = (
    df.groupby(["lat_grid","lon_grid"])
    .size()
    .sort_values(ascending=False)
)

hotspots.head(20)

"""
A. Validation Status Breakdown by Month
"""

# In[]:
pd.crosstab(
    df["month"],
    df["validation_status"],
    margins=True
)

"""
B. Unique Violation Types
"""

# In[]:
import ast

violations = set()

for row in df["violation_type"]:
    try:
        vals = ast.literal_eval(row)
        for v in vals:
            violations.add(v)
    except:
        pass

sorted(violations)

# In[]:
df.groupby("hour").size().sort_index()

"""
Step 1: Create Parking-Only Dataset
"""

# In[]:
parking_df = df[
    df["violation_type"].str.contains(
        "PARKING",
        na=False
    )
].copy()

print("Original:", len(df))
print("Parking:", len(parking_df))
print("Percentage:", round(len(parking_df)/len(df)*100,2))

"""
Step 2: Analyze Parking Violation Distribution
"""

# In[]:
parking_df["violation_type"].value_counts()

"""
Step 3: Check Timestamp Meaning
"""

# In[]:
parking_df.groupby("hour").size().sort_index()

# In[]:
parking_df.groupby("day_of_week").size()

"""
Step 4: Junction Analysis
"""

# In[]:
parking_df["junction_name"].value_counts().head(30)

# In[]:
(
    parking_df["junction_name"] != "No Junction"
).mean()*100

"""
Step 5: Police Station Analysis
"""

# In[]:
parking_df["police_station"].value_counts().head(20)

"""
Step 6: Hotspot Readiness Check
"""

# In[]:
parking_df["lat_grid"] = parking_df["latitude"].round(3)
parking_df["lon_grid"] = parking_df["longitude"].round(3)

grid_counts = (
    parking_df
    .groupby(["lat_grid","lon_grid"])
    .size()
    .sort_values(ascending=False)
)

grid_counts.head(20)

"""
Step 7: Persistence Analysis
"""

# In[]:
parking_df["year_month"] = (
    parking_df["created_datetime"]
    .dt.to_period("M")
)

# In[]:
station_persistence = (
    parking_df
    .groupby("police_station")["year_month"]
    .nunique()
    .sort_values(ascending=False)
)

station_persistence.head(20)

"""
Step 8: Validation Status Investigation
"""

# In[]:
parking_df["validation_status"].value_counts(dropna=False)

# In[]:
pd.crosstab(
    parking_df["month"],
    parking_df["validation_status"],
    margins=True
)

"""
Step 9: Multi-Violation Analysis
"""

# In[]:
import ast

parking_df["num_violations"] = (
    parking_df["violation_type"]
    .apply(lambda x: len(ast.literal_eval(x)))
)

parking_df["num_violations"].value_counts()

"""
Step 10: Determine DBSCAN Parameters
"""

# In[]:
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np

coords = parking_df[
    ["latitude","longitude"]
].dropna()

neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(coords)

distances, indices = neighbors_fit.kneighbors(coords)

distances = np.sort(distances[:,4])

plt.figure(figsize=(10,5))
plt.plot(distances)
plt.title("K-distance plot")
plt.show()

# In[]:


# In[]:

