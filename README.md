# GridLock Patrol Intelligence 🚀
### AI-Driven Parking Violation Hotspot Analysis & Patrol Optimization

GridLock Patrol Intelligence is a data science pipeline and interactive supervisor dashboard that identifies, trends, and prioritizes geographic parking violation hotspots. By converting dense violation coordinates into ranked patrol priority zones with optimized schedules, the system enables traffic enforcement agencies to deploy resources where they are needed most.

---

## 📋 Table of Contents
1. [System Architecture](#-system-architecture)
2. [Repository Structure](#-repository-structure)
3. [Installation & Setup](#-installation--setup)
4. [Running the Data Pipeline](#-running-the-data-pipeline)
5. [Running the Dashboard](#-running-the-dashboard)
6. [Testing & Verification](#-testing--verification)

---

## 🏗 System Architecture

The project consists of a four-stage data processing pipeline followed by an interactive dashboard application. 

```
[Raw Violation Log] 
       │
       ▼
 1. 01_EDA ──────────► Filters Parking Violations (45.4% of total)
       │
       ▼
 2. 02_Engineering ──► Mapped Weights & DBSCAN Spatial Clustering (hotspots)
       │
       ▼
 3. 03_Trajectory ───► Monthly regression trends (Slopes) & Surge Anomaly (Z-Score)
       │
       ▼
 4. 04_Patrol ───────► Combines stats, trends, and surges into Patrol Priority Plan
       │
       ▼
 [05_Dashboard.py] ──► Mapbox Centroids, Leaderboards, Heatmaps & Simulator
```

---

## 📂 Repository Structure

* **`data/`**
  * `dataset.csv`: Raw parking violation database (excluded from Git tracking due to size).
* **Notebooks / Converted Scripts:**
  * `01_EDA.ipynb` / [01_EDA.py](01_EDA.py): Filters and cleans raw parking data.
  * `02_Hotspot_Engineering.ipynb` / [02_Hotspot_Engineering.py](02_Hotspot_Engineering.py): Performs DBSCAN clustering and calculates base priority scores.
  * `03_Hotspot_Trajectory_Analysis.ipynb` / [03_Hotspot_Trajectory_Analysis.py](03_Hotspot_Trajectory_Analysis.py): Computes linear regression trends and latest month anomaly Z-scores.
  * `04_Patrol_Optimizer.ipynb` / [04_Patrol_Optimizer.py](04_Patrol_Optimizer.py): Merges metrics and generates optimized schedules (day of week/hour of day).
* **Dashboard App:**
  * [05_Dashboard.py](05_Dashboard.py): Streamlit dashboard containing Mapbox maps, timeline trends, heatmaps, rosters, and an enforcement Impact Simulator.
* **Pipeline Outputs:**
  * `hotspot_clustered.csv`: Row-level coordinates labeled with DBSCAN cluster IDs.
  * `hotspot_cluster_stats.csv`: Aggregated cluster metrics (severity, police station, centroids).
  * `trajectory_analysis.csv`: Linear regression slope metrics and trend labels.
  * `anomaly_analysis.csv`: Z-score metrics and surge flags for the latest month.
  * `patrol_action_plan.csv`: Final enforcement roster listing patrol windows and deployment levels.
* **Project Metadata:**
  * [README.md](README.md): This file.
  * [DOCUMENTATION.md](DOCUMENTATION.md): Deep-dive document of notebook steps and simulator logic.
  * [requirements.txt](requirements.txt): List of dependencies.
  * [.gitignore](.gitignore): Git ignore rules.

---

## ⚙ Installation & Setup

Ensure you have **Python 3.8+** installed.

### 1. Clone the Repository
```bash
git clone https://github.com/Anurag1734/parking-violation-analysis.git
cd parking-violation-analysis
```

### 2. Set Up a Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```
**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔄 Running the Data Pipeline

To process raw data and regenerate the analytics databases, execute the python scripts in order:

```bash
# Step 1: Clean and isolate parking data
python 01_EDA.py

# Step 2: Run DBSCAN spatial clustering and compute cluster priority
python 02_Hotspot_Engineering.py

# Step 3: Run regression trend fitting and anomaly surge detection
python 03_Hotspot_Trajectory_Analysis.py

# Step 4: Run optimal scheduling and compile the final Patrol Action Plan
python 04_Patrol_Optimizer.py
```
*(Alternatively, you can run the corresponding `.ipynb` notebooks sequentially using Jupyter Lab/Notebook or VS Code.)*

---

## 📊 Running the Dashboard

To launch the interactive supervisor control center, run:

```bash
streamlit run 05_Dashboard.py
```
The application will open automatically in your browser at `http://localhost:8501`.

---

## 🧪 Testing & Verification

To verify that the system is fully operational and correct, follow these manual and programmatic checks:

### 1. Pipeline Verification
Run the pipeline and verify the output files exist and match expected dimensions:

```powershell
# Verify files are generated
Test-Path hotspot_clustered.csv            # Expected: True
Test-Path hotspot_cluster_stats.csv        # Expected: True
Test-Path trajectory_analysis.csv          # Expected: True
Test-Path anomaly_analysis.csv            # Expected: True
Test-Path patrol_action_plan.csv           # Expected: True
```

You can check output row counts using Python:
```bash
python -c "import pandas as pd; print('Plan rows:', len(pd.read_csv('patrol_action_plan.csv')))"
```
*(Expected: 20 rows corresponding to the top 20 enforcement zones.)*

### 2. Dashboard Interface Check
When the Streamlit dashboard opens, verify the following:
* **KPI Cards:** Display numbers for total hotspots, critical hotspots, average severity, and patrol zones.
* **Risk Map:** Mapbox plot renders active hotspots sized by volume and colored by risk.
* **Timeline Charts:** Line plots display historical trends, and the heatmap displays clear peak hours (e.g., peak violations).
* **Patrol Roster:** Table lists recommended day/hour windows and priority rankings. Filter by police station to verify localized views.
* **Watchlist:** Surges are displayed and sorted by Z-score.
* **Impact Simulator:** Move the reduction slider (e.g., to 20%) and verify that the "Simulated Risk Distribution" chart adjusts dynamically without breaking.
