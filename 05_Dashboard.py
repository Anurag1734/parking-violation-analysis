import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Gridlock Patrol Intelligence",
    page_icon="P",
    layout="wide",
)


RISK_ORDER = ["Low", "Medium", "High", "Critical"]
RISK_COLORS = {
    "Low": "#7FB069",
    "Medium": "#F2C14E",
    "High": "#F78154",
    "Critical": "#D7263D",
}
TRAJECTORY_COLORS = {
    "Escalating": "#D7263D",
    "Stable": "#3B82F6",
    "Declining": "#6B7280",
    "Insufficient History": "#9CA3AF",
}


@st.cache_data
def load_data():
    hotspots = pd.read_csv("hotspot_clustered.csv")
    cluster_stats = pd.read_csv("hotspot_cluster_stats.csv")
    trajectory = pd.read_csv("trajectory_analysis.csv")
    anomalies = pd.read_csv("anomaly_analysis.csv")
    patrol_plan = pd.read_csv("patrol_action_plan.csv")
    top_windows = pd.read_csv("best_patrol_windows_detailed.csv")

    hotspots = hotspots[hotspots["cluster_id"] >= 0].copy()
    hotspots["created_datetime"] = pd.to_datetime(hotspots["created_datetime"])
    hotspots["year_month"] = hotspots["created_datetime"].dt.to_period("M").astype(str)

    for frame in [cluster_stats, trajectory, anomalies, patrol_plan, top_windows]:
        frame["cluster_id"] = frame["cluster_id"].astype(float)

    cluster_master = (
        cluster_stats
        .merge(trajectory, on="cluster_id", how="left")
        .merge(anomalies, on="cluster_id", how="left")
        .merge(
            patrol_plan[
                [
                    "cluster_id",
                    "patrol_priority",
                    "recommended_windows",
                    "deployment_level",
                ]
            ],
            on="cluster_id",
            how="left",
        )
    )

    monthly_trends = (
        hotspots
        .groupby(["cluster_id", "year_month"])
        .size()
        .reset_index(name="violations")
    )

    risk_counts = (
        cluster_master["risk_level"]
        .value_counts()
        .reindex(RISK_ORDER, fill_value=0)
        .rename_axis("risk_level")
        .reset_index(name="hotspots")
    )

    watchlist = (
        cluster_master[cluster_master["anomaly_status"] == "Abnormal Surge"]
        .sort_values(["z_score", "patrol_priority"], ascending=False)
        .copy()
    )

    return hotspots, cluster_master, monthly_trends, risk_counts, watchlist, patrol_plan, top_windows


def style_metric(label: str, value: str, delta: str | None = None):
    st.metric(label, value, delta=delta)


def priority_components_after_reduction(cluster_df: pd.DataFrame, reduction_pct: float) -> pd.DataFrame:
    simulated = cluster_df.copy()
    reduction_factor = 1 - reduction_pct / 100.0

    mask_critical = simulated["risk_level"] == "Critical"
    simulated["sim_violations"] = simulated["violations"]
    simulated.loc[mask_critical, "sim_violations"] = (
        simulated.loc[mask_critical, "violations"] * reduction_factor
    )

    simulated["sim_log_violations"] = np.log1p(simulated["sim_violations"])
    log_min = simulated["sim_log_violations"].min()
    log_max = simulated["sim_log_violations"].max()
    denom = log_max - log_min
    if denom == 0:
        simulated["sim_log_violations_norm"] = 0.0
    else:
        simulated["sim_log_violations_norm"] = (
            (simulated["sim_log_violations"] - log_min) / denom
        )

    simulated["sim_priority_score"] = (
        0.45 * simulated["sim_log_violations_norm"]
        + 0.20 * simulated["avg_severity_norm"]
        + 0.10 * simulated["avg_vehicle_weight_norm"]
        + 0.10 * simulated["avg_junction_weight_norm"]
        + 0.10 * simulated["active_months_norm"]
        + 0.05 * simulated["avg_num_violations_norm"]
    )

    q1, q2, q3 = cluster_df["priority_score"].quantile([0.25, 0.50, 0.75]).tolist()
    simulated["sim_risk_level"] = pd.cut(
        simulated["sim_priority_score"],
        bins=[-np.inf, q1, q2, q3, np.inf],
        labels=["Low", "Medium", "High", "Critical"],
        include_lowest=True,
    )
    return simulated


hotspots, cluster_master, monthly_trends, risk_counts, watchlist, patrol_plan, top_windows = load_data()

stations = ["All"] + sorted(cluster_master["top_station"].dropna().unique().tolist())
selected_station = st.sidebar.selectbox("Police Station", stations, index=0)
selected_risks = st.sidebar.multiselect(
    "Risk Levels",
    RISK_ORDER,
    default=RISK_ORDER,
)

filtered_clusters = cluster_master.copy()
if selected_station != "All":
    filtered_clusters = filtered_clusters[filtered_clusters["top_station"] == selected_station]
filtered_clusters = filtered_clusters[filtered_clusters["risk_level"].isin(selected_risks)]

cluster_choices = (
    filtered_clusters
    .sort_values("patrol_priority", ascending=False)["cluster_id"]
    .astype(int)
    .tolist()
)
default_cluster = cluster_choices[0] if cluster_choices else int(cluster_master["cluster_id"].iloc[0])
selected_cluster = st.sidebar.selectbox(
    "Focus Cluster",
    cluster_choices if cluster_choices else cluster_master["cluster_id"].astype(int).tolist(),
    index=0,
)

st.title("Gridlock Patrol Intelligence Dashboard")
st.caption(
    "AI-driven parking intelligence for targeted enforcement: hotspot risk, trajectory, anomaly watchlist, and patrol deployment."
)

metric_cols = st.columns(5)
with metric_cols[0]:
    style_metric("Active Hotspots", f"{len(filtered_clusters):,}")
with metric_cols[1]:
    style_metric(
        "Critical Hotspots",
        f"{(filtered_clusters['risk_level'] == 'Critical').sum():,}",
    )
with metric_cols[2]:
    style_metric(
        "Escalating Hotspots",
        f"{(filtered_clusters['trajectory'] == 'Escalating').sum():,}",
    )
with metric_cols[3]:
    style_metric(
        "Abnormal Surges",
        f"{(filtered_clusters['anomaly_status'] == 'Abnormal Surge').sum():,}",
    )
with metric_cols[4]:
    style_metric(
        "Immediate Enforcement",
        f"{(filtered_clusters['deployment_level'] == 'Immediate Enforcement').sum():,}",
    )

tab_map, tab_risk, tab_trend, tab_watch, tab_patrol, tab_impact = st.tabs(
    [
        "Hotspot Map",
        "Risk Zones",
        "Trajectory View",
        "Watchlist",
        "Patrol Recommendations",
        "Impact Simulator",
    ]
)

with tab_map:
    st.subheader("DBSCAN Hotspot Map")
    map_df = filtered_clusters.dropna(subset=["centroid_lat", "centroid_lon"]).copy()
    map_df["cluster_label"] = map_df["cluster_id"].astype(int).astype(str)
    fig_map = px.scatter_mapbox(
        map_df,
        lat="centroid_lat",
        lon="centroid_lon",
        color="risk_level",
        size="violations",
        hover_name="cluster_label",
        hover_data={
            "top_station": True,
            "trajectory": True,
            "anomaly_status": True,
            "patrol_priority": ":.3f",
            "centroid_lat": False,
            "centroid_lon": False,
        },
        color_discrete_map=RISK_COLORS,
        zoom=10,
        height=620,
    )
    fig_map.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=0, r=0, t=0, b=0),
        legend_title_text="Risk Zone",
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.dataframe(
        map_df[
            [
                "cluster_id",
                "top_station",
                "risk_level",
                "trajectory",
                "anomaly_status",
                "priority_score",
                "patrol_priority",
                "deployment_level",
            ]
        ]
        .sort_values("patrol_priority", ascending=False)
        .reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

with tab_risk:
    st.subheader("Risk Zones")
    left, right = st.columns([1.2, 1])

    with left:
        risk_fig = px.bar(
            risk_counts,
            x="risk_level",
            y="hotspots",
            color="risk_level",
            color_discrete_map=RISK_COLORS,
            category_orders={"risk_level": RISK_ORDER},
            text="hotspots",
            height=420,
        )
        risk_fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Hotspots")
        st.plotly_chart(risk_fig, use_container_width=True)

    with right:
        top_risk_table = (
            filtered_clusters[
                [
                    "cluster_id",
                    "top_station",
                    "violations",
                    "priority_score",
                    "risk_level",
                ]
            ]
            .sort_values(["risk_level", "priority_score"], ascending=[False, False])
            .head(15)
            .reset_index(drop=True)
        )
        st.dataframe(top_risk_table, use_container_width=True, hide_index=True)

with tab_trend:
    st.subheader("Trajectory View")
    focus = cluster_master[cluster_master["cluster_id"].astype(int) == int(selected_cluster)].iloc[0]
    trend_df = (
        monthly_trends[monthly_trends["cluster_id"].astype(int) == int(selected_cluster)]
        .sort_values("year_month")
        .copy()
    )

    trend_cols = st.columns(4)
    with trend_cols[0]:
        style_metric("Trajectory", focus["trajectory"])
    with trend_cols[1]:
        style_metric("Slope", f"{focus['slope']:.2f}")
    with trend_cols[2]:
        style_metric("R2 Confidence", f"{0 if pd.isna(focus['r2']) else focus['r2']:.2f}")
    with trend_cols[3]:
        style_metric("Anomaly", focus["anomaly_status"])

    trend_fig = go.Figure()
    trend_fig.add_trace(
        go.Scatter(
            x=trend_df["year_month"],
            y=trend_df["violations"],
            mode="lines+markers",
            line=dict(color=TRAJECTORY_COLORS.get(focus["trajectory"], "#3B82F6"), width=4),
            marker=dict(size=10),
            name="Monthly Violations",
        )
    )
    trend_fig.update_layout(
        height=420,
        xaxis_title="Month",
        yaxis_title="Violations",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(trend_fig, use_container_width=True)

    st.dataframe(
        trend_df.rename(columns={"year_month": "month"}),
        use_container_width=True,
        hide_index=True,
    )

with tab_watch:
    st.subheader("Watchlist: Abnormal Surges")
    st.caption("Clusters where the latest complete month materially exceeded the cluster's baseline.")

    watch_cols = st.columns(3)
    with watch_cols[0]:
        style_metric("Watchlist Size", f"{len(watchlist):,}")
    with watch_cols[1]:
        style_metric("Max Z-Score", f"{watchlist['z_score'].max():.2f}")
    with watch_cols[2]:
        style_metric(
            "Critical In Watchlist",
            f"{(watchlist['risk_level'] == 'Critical').sum():,}",
        )

    watchlist_table = watchlist[
        [
            "cluster_id",
            "top_station",
            "risk_level",
            "trajectory",
            "z_score",
            "current_month",
            "patrol_priority",
            "deployment_level",
        ]
    ].reset_index(drop=True)
    st.dataframe(watchlist_table, use_container_width=True, hide_index=True)

with tab_patrol:
    st.subheader("Patrol Recommendations")
    top20 = patrol_plan[
        [
            "cluster_id",
            "top_station",
            "risk_level",
            "trajectory",
            "anomaly_status",
            "patrol_priority",
            "deployment_level",
            "recommended_windows",
        ]
    ].head(20)

    lead_left, lead_right = st.columns([1.15, 1])
    with lead_left:
        st.markdown("**Top 20 Action Plan**")
        st.dataframe(top20, use_container_width=True, hide_index=True)
    with lead_right:
        st.markdown("**Top Patrol Windows**")
        focus_windows = top_windows[top_windows["cluster_id"].astype(int) == int(selected_cluster)].copy()
        focus_windows["hour_share_pct"] = (100 * focus_windows["hour_percentage"]).round(1)
        focus_windows = focus_windows.sort_values("window_score", ascending=False)
        st.dataframe(
            focus_windows[
                [
                    "day_of_week",
                    "hour",
                    "window_violations",
                    "hour_share_pct",
                    "window_score",
                ]
            ].head(10),
            use_container_width=True,
            hide_index=True,
        )

with tab_impact:
    st.subheader("Impact Simulator")
    st.caption("If targeted enforcement reduces violations in current Critical hotspots, how many fall out of the Critical risk zone?")

    reduction_pct = st.slider(
        "Simulated violation reduction for current Critical hotspots",
        min_value=0,
        max_value=50,
        value=20,
        step=5,
    )

    simulated = priority_components_after_reduction(cluster_master, reduction_pct)
    original_critical = simulated[simulated["risk_level"] == "Critical"].copy()
    moved_to_high_or_lower = (original_critical["sim_risk_level"] != "Critical").sum()
    remaining_critical = (original_critical["sim_risk_level"] == "Critical").sum()

    impact_cols = st.columns(4)
    with impact_cols[0]:
        style_metric("Current Critical", f"{len(original_critical):,}")
    with impact_cols[1]:
        style_metric("Still Critical", f"{remaining_critical:,}")
    with impact_cols[2]:
        style_metric("Become High or Lower", f"{moved_to_high_or_lower:,}")
    with impact_cols[3]:
        pct = 0 if len(original_critical) == 0 else 100 * moved_to_high_or_lower / len(original_critical)
        style_metric("Critical Reduction", f"{pct:.1f}%")

    impact_fig = px.histogram(
        simulated,
        x="sim_risk_level",
        category_orders={"sim_risk_level": RISK_ORDER},
        color="sim_risk_level",
        color_discrete_map=RISK_COLORS,
        title="Simulated Risk Distribution After Enforcement Impact",
    )
    impact_fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Hotspots")
    st.plotly_chart(impact_fig, use_container_width=True)

    impact_table = original_critical[
        [
            "cluster_id",
            "top_station",
            "priority_score",
            "sim_priority_score",
            "risk_level",
            "sim_risk_level",
        ]
    ].sort_values("sim_priority_score", ascending=False)
    st.dataframe(impact_table, use_container_width=True, hide_index=True)
