from IPython.display import display

"""
Step 0 — Load Data
"""

# In[]:
import pandas as pd
import numpy as np
import ast

df = pd.read_csv("data/dataset.csv")

print(df.shape)
df.head()

"""
Step 1 — Apply Validation Filter
"""

# In[]:
valid_df = df[
    df["validation_status"].isna() |
    (df["validation_status"] == "approved")
].copy()

print("Original:", len(df))
print("Filtered:", len(valid_df))

"""
Step 2 — Parse Timestamps
"""

# In[]:
date_cols = [
    "created_datetime",
    "modified_datetime",
    "validation_timestamp"
]

for col in date_cols:
    if col in valid_df.columns:
        valid_df[col] = pd.to_datetime(
            valid_df[col],
            errors="coerce",
            utc=True
        ).dt.tz_localize(None)

valid_df[date_cols].head()

"""
Step 3 — Parse Violation Lists
"""

# In[]:
valid_df["violations_list"] = valid_df[
    "violation_type"
].apply(
    lambda x: ast.literal_eval(x)
    if pd.notna(x)
    else []
)

# In[]:
valid_df[
    ["violation_type","violations_list"]
].head()

"""
Step 4 — Create Number of Violations Feature
"""

# In[]:
valid_df["num_violations"] = (
    valid_df["violations_list"]
    .apply(len)
)

valid_df["num_violations"].value_counts()

"""
Step 5 — Severity Mapping
"""

# In[]:
severity_map = {
    "NO PARKING":1,
    "WRONG PARKING":2,
    "PARKING ON FOOTPATH":3,
    "DOUBLE PARKING":4,
    "PARKING IN A MAIN ROAD":5,
    "PARKING NEAR ROAD CROSSING":6,
    "PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS":7
}

# In[]:
valid_df["max_severity"] = (
    valid_df["violations_list"]
    .apply(
        lambda v:
        max(
            [severity_map.get(x,1) for x in v]
        )
        if len(v)>0
        else 1
    )
)

valid_df["max_severity"].describe()

"""
Step 6 — Vehicle Weight Mapping
"""

# In[]:
vehicle_weight = {
    "SCOOTER":1.0,
    "MOPED":1.0,
    "MOTOR CYCLE":1.0,

    "CAR":1.5,
    "JEEP":1.5,
    "OTHERS":1.5,

    "PASSENGER AUTO":1.8,
    "GOODS AUTO":1.8,

    "VAN":2.0,
    "MAXI-CAB":2.0,

    "TEMPO":2.5,
    "LGV":2.5,

    "MINI LORRY":3.0,
    "TRACTOR":3.0,

    "PRIVATE BUS":3.5,
    "TOURIST BUS":3.5,
    "SCHOOL VEHICLE":3.5,
    "FACTORY BUS":3.5,
    "BUS (BMTC/KSRTC)":3.5,

    "HGV":4.0,
    "TANKER":4.0,
    "LORRY/GOODS VEHICLE":4.0
}

# In[]:
valid_df["vehicle_weight"] = (
    valid_df["vehicle_type"]
    .map(vehicle_weight)
    .fillna(1.5)
)

valid_df[
    ["vehicle_type","vehicle_weight"]
].sample(10)

"""
Step 7 — Junction Weight
"""

# In[]:
junction_counts = (
    valid_df["junction_name"]
    .value_counts()
)

# Normalize
junction_weight = (
    junction_counts
    /
    junction_counts.max()
)

# Apply
valid_df["junction_weight"] = (
    valid_df["junction_name"]
    .map(junction_weight)
)

# Check
valid_df[
    ["junction_name","junction_weight"]
].head()

# In[]:
# 5 rows are missing junction weights:
valid_df["junction_weight"] = (
    valid_df["junction_weight"]
    .fillna(0)
)

"""
Step 8 — Temporal Features
"""

# In[]:
valid_df["hour"] = (
    valid_df["created_datetime"]
    .dt.hour
)

valid_df["day_of_week"] = (
    valid_df["created_datetime"]
    .dt.day_name()
)

valid_df["month"] = (
    valid_df["created_datetime"]
    .dt.month
)

valid_df["is_weekend"] = (
    valid_df["created_datetime"]
    .dt.dayofweek >= 5
).astype(int)

# Check

valid_df[
    [
        "hour",
        "day_of_week",
        "month",
        "is_weekend"
    ]
].head()

"""
Step 9 — Save Engineered Dataset
"""

# In[]:
valid_df.to_csv(
    "engineered_dataset.csv",
    index=False
)

print("Saved")

"""
# ==========================
# DBSCAN HOTSPOT GENERATION
# ==========================
"""

"""
Step 1 — Extract Coordinates
"""

# In[]:
coords = valid_df[
    ["latitude", "longitude"]
].dropna()

print(coords.shape)

"""
Step 2 — Convert to Radians
"""

# In[]:
import numpy as np

coords_rad = np.radians(coords)

"""
Step 3 — K-Distance Plot
"""

# In[]:
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

neighbors = NearestNeighbors(
    n_neighbors=5,
    metric="haversine"
)

neighbors_fit = neighbors.fit(coords_rad)

distances, indices = neighbors_fit.kneighbors(coords_rad)

distances = np.sort(distances[:, 4])

plt.figure(figsize=(12,5))
plt.plot(distances)
plt.title("K-Distance Plot")
plt.xlabel("Points Sorted by Distance")
plt.ylabel("Distance (Radians)")
plt.grid()
plt.show()

# In[]:
plt.figure(figsize=(12,5))

plt.plot(distances[-10000:])

plt.title("K-Distance Plot (Last 10,000 Points)")
plt.xlabel("Points")
plt.ylabel("Distance (Radians)")
plt.grid()

plt.show()

# In[]:
plt.figure(figsize=(12,5))

plt.plot(distances[-5000:])

plt.title("K-Distance Plot (Last 5,000 Points)")
plt.xlabel("Points")
plt.ylabel("Distance (Radians)")
plt.grid()

plt.show()

# In[]:
print("95% :", np.percentile(distances,95))
print("97% :", np.percentile(distances,97))
print("98% :", np.percentile(distances,98))
print("99% :", np.percentile(distances,99))
print("99.5% :", np.percentile(distances,99.5))
print("99.9% :", np.percentile(distances,99.9))

# In[]:
coords.describe()

# In[]:
# DB Scan parameters

eps = 0.00002627      # 99.5 percentile
min_samples = 20

"""
Run DBScan
"""

# In[]:
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(
    eps=0.00002627,
    min_samples=20,
    metric="haversine",
    algorithm="ball_tree"
)

clusters = dbscan.fit_predict(coords_rad)

valid_df.loc[coords.index, "cluster_id"] = clusters

"""
First Evaluation
"""

# In[]:
n_clusters = len(
    set(clusters)
) - (1 if -1 in clusters else 0)

noise_pct = (
    (clusters == -1).mean()
) * 100

print("Clusters:", n_clusters)
print("Noise %:", round(noise_pct,2))

"""
Cluster Size Analysis
"""

# In[]:
cluster_sizes = (
    valid_df["cluster_id"]
    .value_counts()
    .sort_values(ascending=False)
)

cluster_sizes.head(20)

"""
Hotspot Summary
"""

# In[]:
hotspot_summary = (
    valid_df[valid_df["cluster_id"] != -1]
    .groupby("cluster_id")
    .agg(
        violations=("cluster_id","size"),
        avg_severity=("max_severity","mean"),
        avg_vehicle_weight=("vehicle_weight","mean"),
        avg_junction_weight=("junction_weight","mean")
    )
    .sort_values(
        "violations",
        ascending=False
    )
)

hotspot_summary.head(20)

# In[]:
from sklearn.cluster import DBSCAN

eps_test = 0.00001623

dbscan_test = DBSCAN(
    eps=0.00000962,
    min_samples=20,
    metric="haversine",
    algorithm="ball_tree"
)

clusters_test = dbscan_test.fit_predict(coords_rad)

print("Clusters:",
      len(set(clusters_test)) - (1 if -1 in clusters_test else 0))

print("Noise %:",
      round((clusters_test == -1).mean()*100,2))

# In[]:
cluster_sizes_test = (
    pd.Series(clusters_test)
    .value_counts()
    .sort_values(ascending=False)
)

cluster_sizes_test.head(20)

"""
Freeze DBSCAN
"""

# In[]:
valid_df.loc[coords.index, "cluster_id"] = clusters_test

valid_df.to_csv(
    "hotspot_clustered.csv",
    index=False
)

"""
Step 1 — Cluster Density
"""

# In[]:
cluster_stats = (
    valid_df[valid_df["cluster_id"] != -1]
    .groupby("cluster_id")
    .agg(
        violations=("cluster_id", "size"),
        avg_severity=("max_severity", "mean"),
        avg_vehicle_weight=("vehicle_weight", "mean"),
        avg_junction_weight=("junction_weight", "mean"),
        avg_num_violations=("num_violations", "mean")
    )
)

cluster_stats.head()

"""
Step 2 — Persistence Score
"""

# In[]:
valid_df["year_month"] = (
    valid_df["created_datetime"]
    .dt.to_period("M")
)

# In[]:
persistence = (
    valid_df[valid_df["cluster_id"] != -1]
    .groupby("cluster_id")["year_month"]
    .nunique()
    .rename("active_months")
)

cluster_stats = cluster_stats.join(persistence)

"""
Step 3 — Normalize Features
"""

# In[]:
from sklearn.preprocessing import MinMaxScaler

cluster_stats["log_violations"] = np.log1p(
    cluster_stats["violations"]
)

cols = [
    "violations",
    "avg_severity",
    "avg_vehicle_weight",
    "avg_junction_weight",
    "avg_num_violations",
    "active_months"
]

scaler = MinMaxScaler()

cluster_stats[[c + "_norm" for c in cols]] = (
    scaler.fit_transform(
        cluster_stats[cols]
    )
)

cluster_stats["log_violations_norm"] = (
    MinMaxScaler()
    .fit_transform(
        cluster_stats[["log_violations"]]
    )
)

"""
Step 4 — Priority Score
"""

# In[]:
cluster_stats["priority_score"] = (
    0.45 * cluster_stats["log_violations_norm"]
    + 0.20 * cluster_stats["avg_severity_norm"]
    + 0.10 * cluster_stats["avg_vehicle_weight_norm"]
    + 0.10 * cluster_stats["avg_junction_weight_norm"]
    + 0.10 * cluster_stats["active_months_norm"]
    + 0.05 * cluster_stats["avg_num_violations_norm"]
)

"""
Step 5 — Top Hotspots
"""

# In[]:
cluster_stats.sort_values(
    "priority_score",
    ascending=False
).head(20)

# In[]:
cluster_stats[
[
    "violations",
    "avg_severity",
    "priority_score"
]
].sort_values(
    "priority_score",
    ascending=False
).head(20)

# In[]:
cluster_stats[
[
    "priority_score",
    "violations",
    "avg_junction_weight",
    "avg_severity"
]
].corr()

# In[]:
cluster_centroids = (
    valid_df[valid_df["cluster_id"] != -1]
    .groupby("cluster_id")
    .agg(
        centroid_lat=("latitude","mean"),
        centroid_lon=("longitude","mean")
    )
)
cluster_centroids.head()

# In[]:
top_station = (
    valid_df[valid_df["cluster_id"] != -1]
    .groupby("cluster_id")["police_station"]
    .agg(lambda x: x.mode().iloc[0])
)
top_station.head()

# In[]:
cluster_stats = cluster_stats.drop(
    columns=["centroid_lat","centroid_lon"],
    errors="ignore"
)

cluster_stats = cluster_stats.join(cluster_centroids)

cluster_stats["top_station"] = top_station
cluster_stats.head()

# In[]:
cluster_stats.columns

# In[]:
cluster_stats[
[
    "violations",
    "avg_severity",
    "priority_score",
    "active_months"
]
].describe()

# In[]:
cluster_stats["risk_level"] = pd.qcut(
    cluster_stats["priority_score"],
    q=[0, 0.50, 0.80, 0.95, 1.0],
    labels=[
        "Low",
        "Medium",
        "High",
        "Critical"
    ]
)
cluster_stats["risk_level"].value_counts()

# In[]:
cluster_stats.to_csv(
    "hotspot_cluster_stats.csv"
)

# In[]:
cluster_stats[
[
    "violations",
    "avg_severity",
    "avg_vehicle_weight",
    "avg_junction_weight",
    "avg_num_violations",
    "active_months",
    "risk_level"
]
].head()

# In[]:

