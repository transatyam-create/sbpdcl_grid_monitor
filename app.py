import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(
    page_title="SBPDCL Grid Monitor | Kaimur Circle",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Load Trained Model Bundle
@st.cache_resource
def load_artifacts():
    return joblib.load("sbpdcl_models_v2.joblib")

artifact = load_artifacts()
load_model = artifact["load_model"]
hotspot_model = artifact["hotspot_model"]
health_model = artifact["health_model"]
cause_model = artifact["cause_model"]
blackout_model = artifact["blackout_model"]
features = artifact["features"]
dt_type_to_code = artifact["dt_type_to_code"]
code_to_cause = artifact["code_to_cause"]
dt_metadata = artifact["dt_metadata"]

# Kaimur Circle Substation Coordinates
KAIMUR_COORDS = {
    "Urban_Domestic_Commercial": {"lat": 25.0418, "lon": 83.6075, "substation": "Bhabua 33/11 kV PSS"},
    "Rural_Domestic":            {"lat": 25.0335, "lon": 83.4862, "substation": "Chainpur Rural Feeder"},
    "Agricultural_Tubewell":     {"lat": 25.1711, "lon": 83.6214, "substation": "Mohania Agri Substation"},
    "Industrial_Rice_Mill":      {"lat": 25.1050, "lon": 83.6150, "substation": "Kaimur Industrial Corridor"},
}

# 3. Left Sidebar: Theme Toggle & Direct Input Controls
with st.sidebar:
    col_logo, col_theme = st.columns([1.4, 1])
    with col_logo:
        st.markdown("### **SBPDCL Grid AI**")
    with col_theme:
        dark_mode = st.toggle("Dark mode", value=False)

    st.caption("Kaimur Circle • Distribution Monitoring")
    st.divider()

    st.markdown("**Input Parameters**")

    selected_dt = st.selectbox(
        "Transformer Type",
        options=list(dt_metadata.keys()),
        format_func=lambda x: f"{x.replace('_', ' ')} ({dt_metadata[x]['rated_capacity_kva']} kVA)"
    )

    meta = dt_metadata[selected_dt]
    rated_kva = meta["rated_capacity_kva"]
    pf = meta["power_factor"]

    col_date, col_hour = st.columns([1.3, 1])
    with col_date:
        selected_date = st.date_input(
            "Date",
            value=datetime.date(2026, 6, 15)
        )
    with col_hour:
        hour_of_day = st.number_input(
            "Hour (0–23)",
            min_value=0,
            max_value=23,
            value=16,
            step=1,
            format="%d"
        )

    day_of_year = selected_date.timetuple().tm_yday
    st.caption(f"Day {day_of_year} of year • {selected_date.strftime('%B')} • {hour_of_day:02d}:00 hrs")

    col_load, col_temp = st.columns(2)
    with col_load:
        current_load_kva = st.number_input(
            "Load (kVA)",
            min_value=0.0,
            max_value=float(rated_kva * 1.6),
            value=round(rated_kva * 0.78, 1),
            step=1.0,
            format="%.1f",
            help=f"Rated capacity: {rated_kva} kVA"
        )
    with col_temp:
        ambient_temp = st.number_input(
            "Temp (°C)",
            min_value=0.0,
            max_value=55.0,
            value=39.0,
            step=0.5,
            format="%.1f",
            help="Ambient temperature in Celsius"
        )

    st.caption(f"Rated Capacity: {rated_kva} kVA • Power Factor: {pf}")

    is_holiday_label = st.radio(
        "Holiday or Festival",
        options=["No", "Yes"],
        horizontal=True
    )
    is_holiday = 1 if is_holiday_label == "Yes" else 0

    with st.expander("Past Failure History"):
        past_failures = st.number_input(
            "Past Failures Count",
            min_value=0,
            max_value=100,
            value=meta["total_past_failures"]
        )
        past_blackout_hrs = st.number_input(
            "Past Blackout Hours",
            min_value=0,
            max_value=1000,
            value=meta["total_past_blackout_hours"]
        )

    st.markdown("")
    run_btn = st.button("Run Model", type="primary", use_container_width=True)

# 4. High-Contrast Dynamic Theme Styling
if dark_mode:
    bg_color = "#0e1117"
    sidebar_bg = "#161b22"
    card_bg = "#161b22"
    input_bg = "#21262d"
    border_color = "#30363d"
    text_primary = "#f8fafc"
    text_secondary = "#94a3b8"
    grid_color = "#262c36"
    chart_line = "#60a5fa"
    chart_fill = "rgba(96, 165, 250, 0.18)"
    plotly_template = "plotly_dark"
    map_style = "carto-darkmatter"
    alert_text_color = "#f8fafc"
else:
    bg_color = "#f4f6f9"
    sidebar_bg = "#ffffff"
    card_bg = "#ffffff"
    input_bg = "#f8fafc"
    border_color = "#cbd5e1"
    text_primary = "#0f172a"
    text_secondary = "#334155"
    grid_color = "#e2e8f0"
    chart_line = "#1d4ed8"
    chart_fill = "rgba(29, 78, 216, 0.14)"
    plotly_template = "plotly_white"
    map_style = "carto-positron"
    alert_text_color = "#0f172a"

st.markdown(f"""
<style>
    .stApp, [data-testid="stHeader"] {{
        background-color: {bg_color} !important;
        color: {text_primary} !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        border-right: 1px solid {border_color};
    }}
    [data-testid="stSidebar"] * {{
        color: {text_primary} !important;
    }}
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] div[data-baseweb="select"] > div {{
        background-color: {input_bg} !important;
        color: {text_primary} !important;
        border-color: {border_color} !important;
    }}
    [data-testid="stSidebar"] button[kind="primary"] {{
        background-color: #1d4ed8 !important;
        border-color: #1d4ed8 !important;
    }}
    [data-testid="stSidebar"] button[kind="primary"] * {{
        color: #ffffff !important;
    }}
    [data-testid="stAlert"] * {{
        color: {alert_text_color} !important;
    }}
    .kpi-card {{
        background-color: {card_bg};
        border: 1px solid {border_color};
        border-radius: 8px;
        padding: 12px 14px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
    }}
    .kpi-title {{
        font-size: 0.78rem;
        color: {text_secondary};
        font-weight: 600;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    .kpi-value {{
        font-size: 1.35rem;
        font-weight: 700;
        color: {text_primary};
        line-height: 1.2;
    }}
    .kpi-sub {{
        font-size: 0.75rem;
        color: {text_secondary};
        font-weight: 500;
        margin-top: 3px;
    }}
    .ticker-bar {{
        background-color: {card_bg};
        border: 1px solid {border_color};
        border-radius: 6px;
        padding: 9px 14px;
        font-size: 0.83rem;
        color: {text_primary};
        margin: 10px 0 14px 0;
    }}
    .section-heading {{
        font-size: 1.05rem;
        font-weight: 700;
        color: {text_primary};
    }}
    .section-sub {{
        font-size: 0.80rem;
        color: {text_secondary};
        font-weight: 500;
    }}
</style>
""", unsafe_allow_html=True)

# Helper Function to Build Feature Row
def build_feature_row(dt_name, target_dt, hr, load_kva, temp_c, holiday_flag, p_fail, p_bo):
    m = dt_metadata[dt_name]
    r_kva = m["rated_capacity_kva"]
    p_fac = m["power_factor"]
    doy = target_dt.timetuple().tm_yday
    dow = target_dt.weekday()
    wknd = 1 if dow >= 5 else 0
    mo = target_dt.month

    return pd.DataFrame([{
        "dt_type_code": dt_type_to_code[dt_name],
        "rated_capacity_kva": r_kva,
        "power_factor": p_fac,
        "hour_of_day": hr,
        "hour_sin": np.sin(2 * np.pi * hr / 24.0),
        "hour_cos": np.cos(2 * np.pi * hr / 24.0),
        "day_of_year": doy,
        "day_sin": np.sin(2 * np.pi * doy / 365.0),
        "day_cos": np.cos(2 * np.pi * doy / 365.0),
        "day_of_week": dow,
        "is_weekend": wknd,
        "is_holiday": holiday_flag,
        "month": mo,
        "ambient_temperature_c": temp_c,
        "unconstrained_demand_kva": load_kva,
        "overload_ratio": load_kva / r_kva,
        "past_failures_count": p_fail,
        "past_blackout_hours": p_bo
    }])[features]

if "has_run" not in st.session_state:
    st.session_state.has_run = True
if run_btn:
    st.session_state.has_run = True

# Run Predictions
base_dt = datetime.datetime.combine(selected_date, datetime.time(hour=int(hour_of_day)))
input_row = build_feature_row(
    selected_dt, base_dt, int(hour_of_day), current_load_kva, ambient_temp, is_holiday, past_failures, past_blackout_hrs
)

pred_next_load_kva = float(load_model.predict(input_row)[0])
pred_next_load_kw = pred_next_load_kva * pf
load_delta_pct = ((pred_next_load_kva - current_load_kva) / max(current_load_kva, 0.1)) * 100
overload_ratio = current_load_kva / rated_kva

pred_hotspot_c = float(hotspot_model.predict(input_row)[0])
pred_health_tier = int(health_model.predict(input_row)[0])
health_probs = health_model.predict_proba(input_row)[0]
pred_cause_label = code_to_cause[int(cause_model.predict(input_row)[0])].replace("_", " ")
pred_blackout_hours = max(6.0, float(blackout_model.predict(input_row)[0]))

health_score = int(np.clip(
    (health_probs[0] * 100) - max(0, (overload_ratio - 0.8) * 60) - max(0, (pred_hotspot_c - 80) * 0.8),
    5, 100
))

status_map = {
    0: ("Healthy", "#059669" if not dark_mode else "#10b981"),
    1: ("Warning", "#d97706" if not dark_mode else "#f59e0b"),
    2: ("Critical", "#dc2626" if not dark_mode else "#ef4444")
}
status_text, status_color = status_map[pred_health_tier]

# 5. Top 7-Card KPI Strip
k1, k2, k3, k4, k5, k6, k7 = st.columns(7)

with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title"><span style="color:#2563eb;">●</span> Current Load</div>
        <div class="kpi-value">{current_load_kva:.1f} <span style="font-size:0.8rem;font-weight:600;">kVA</span></div>
        <div class="kpi-sub">{current_load_kva * pf:.1f} kW active</div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title"><span style="color:#059669;">●</span> Next-Hour Forecast</div>
        <div class="kpi-value">{pred_next_load_kva:.1f} <span style="font-size:0.8rem;font-weight:600;">kVA</span></div>
        <div class="kpi-sub">{load_delta_pct:+.1f}% from current</div>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title"><span style="color:#d97706;">●</span> Capacity Used</div>
        <div class="kpi-value">{overload_ratio*100:.1f}<span style="font-size:0.8rem;font-weight:600;">%</span></div>
        <div class="kpi-sub">Rated: {rated_kva} kVA</div>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title"><span style="color:#dc2626;">●</span> Winding Temp</div>
        <div class="kpi-value">{pred_hotspot_c:.1f} <span style="font-size:0.8rem;font-weight:600;">°C</span></div>
        <div class="kpi-sub">Ambient: {ambient_temp:.1f}°C</div>
    </div>""", unsafe_allow_html=True)

with k5:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title"><span style="color:{status_color};">●</span> Health Score</div>
        <div class="kpi-value">{health_score}<span style="font-size:0.8rem;font-weight:600;">/100</span></div>
        <div class="kpi-sub" style="color:{status_color};font-weight:700;">{status_text}</div>
    </div>""", unsafe_allow_html=True)

with k6:
    bo_val = f"{pred_blackout_hours:.1f} hrs" if pred_health_tier == 2 else "0 hrs"
    bo_sub = "Estimated outage" if pred_health_tier == 2 else "No outage expected"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title"><span style="color:#7c3aed;">●</span> Blackout Risk</div>
        <div class="kpi-value">{bo_val}</div>
        <div class="kpi-sub">{bo_sub}</div>
    </div>""", unsafe_allow_html=True)

with k7:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title"><span style="color:#0284c7;">●</span> Day of Year</div>
        <div class="kpi-value">Day {day_of_year}</div>
        <div class="kpi-sub">{'Holiday' if is_holiday else 'Regular Day'}</div>
    </div>""", unsafe_allow_html=True)

# 6. Summary Status Bar
st.markdown(f"""
<div class="ticker-bar">
    <b>Feeder Summary ({meta['dt_id']} • {meta['location'].replace('_', ' ')}):</b> &nbsp;
    Condition: <b>{pred_cause_label}</b> &nbsp;|&nbsp;
    Past Record: <b>{past_failures} Failures, {past_blackout_hrs} Blackout Hrs</b> &nbsp;|&nbsp;
    Failure Risk: <b>{health_probs[2]*100:.1f}%</b>
</div>
""", unsafe_allow_html=True)

# 7. Main Two-Column Layout: Left = 48h Demand Chart | Right = Substation Map & Health Summary
left_panel, right_panel = st.columns([1.35, 1])

with left_panel:
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:6px;">
        <span class="section-heading">Load Curve — Last 24 Hours and Next 24 Hours</span>
        <span class="section-sub">Hourly model • Rated capacity {rated_kva} kVA</span>
    </div>
    """, unsafe_allow_html=True)

    times_past, loads_past = [], []
    times_future, loads_future = [], []

    for offset in range(-24, 1):
        t_step = base_dt + datetime.timedelta(hours=offset)
        temp_step = ambient_temp - 4.0 * np.cos(2 * np.pi * (t_step.hour - 14) / 24.0) + 4.0 * np.cos(2 * np.pi * (int(hour_of_day) - 14) / 24.0)
        if offset == 0:
            est_load = current_load_kva
        else:
            row_step = build_feature_row(
                selected_dt, t_step, t_step.hour, current_load_kva, temp_step, is_holiday, past_failures, past_blackout_hrs
            )
            est_load = float(load_model.predict(row_step)[0])
        times_past.append(t_step)
        loads_past.append(est_load)

    times_future.append(base_dt)
    loads_future.append(current_load_kva)
    rolling_load = current_load_kva

    for offset in range(1, 25):
        t_step = base_dt + datetime.timedelta(hours=offset)
        temp_step = ambient_temp - 4.0 * np.cos(2 * np.pi * (t_step.hour - 14) / 24.0) + 4.0 * np.cos(2 * np.pi * (int(hour_of_day) - 14) / 24.0)
        row_step = build_feature_row(
            selected_dt, t_step, t_step.hour, rolling_load, temp_step, is_holiday, past_failures, past_blackout_hrs
        )
        rolling_load = float(load_model.predict(row_step)[0])
        times_future.append(t_step)
        loads_future.append(rolling_load)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=times_past,
        y=loads_past,
        mode="lines",
        name="Last 24 Hours",
        line=dict(color=chart_line, width=2.4),
        fill="tozeroy",
        fillcolor=chart_fill
    ))

    fig.add_trace(go.Scatter(
        x=times_future,
        y=loads_future,
        mode="lines",
        name="Forecast (Next 24 Hours)",
        line=dict(color=chart_line, width=2.4, dash="dash")
    ))

    fig.add_trace(go.Scatter(
        x=[base_dt],
        y=[current_load_kva],
        mode="markers",
        name="Selected Time",
        marker=dict(color=chart_line, size=9, line=dict(color=card_bg, width=1.5))
    ))

    fig.add_vline(
        x=base_dt.timestamp() * 1000,
        line_dash="dot",
        line_color=text_secondary,
        annotation_text=f"{int(hour_of_day):02d}:00",
        annotation_position="top",
        annotation_font=dict(color=text_primary, size=12)
    )

    fig.add_hline(
        y=rated_kva,
        line_dash="dash",
        line_color="#dc2626",
        annotation_text=f"Rated Limit ({rated_kva} kVA)",
        annotation_position="bottom right",
        annotation_font=dict(color="#dc2626", size=12)
    )

    fig.update_layout(
        template=plotly_template,
        height=390,
        margin=dict(l=12, r=15, t=35, b=12),
        font=dict(color=text_primary, size=12),
        xaxis=dict(
            tickfont=dict(color=text_primary, size=11),
            gridcolor=grid_color,
            linecolor=border_color
        ),
        yaxis=dict(
            title=dict(text="Load (kVA)", font=dict(color=text_primary, size=12)),
            tickfont=dict(color=text_primary, size=11),
            gridcolor=grid_color,
            linecolor=border_color
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(color=text_primary, size=12)
        ),
        paper_bgcolor=card_bg,
        plot_bgcolor=card_bg,
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

with right_panel:
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:6px;">
        <span class="section-heading">Substation Location</span>
        <span class="section-sub">Kaimur Grid</span>
    </div>
    """, unsafe_allow_html=True)

    map_rows = []
    for dt_k, coord in KAIMUR_COORDS.items():
        is_selected = (dt_k == selected_dt)
        map_rows.append({
            "Feeder": coord["substation"],
            "Type": dt_k.replace("_", " "),
            "lat": coord["lat"],
            "lon": coord["lon"],
            "Capacity": f"{dt_metadata[dt_k]['rated_capacity_kva']} kVA",
            "Size": 20 if is_selected else 11,
            "Color": status_color if is_selected else "#475569",
            "Status": f"Selected: {status_text}" if is_selected else "Monitored"
        })
    map_df = pd.DataFrame(map_rows)

    map_fig = go.Figure(go.Scattermap(
        lat=map_df["lat"],
        lon=map_df["lon"],
        mode="markers+text",
        marker=dict(size=map_df["Size"], color=map_df["Color"]),
        text=map_df["Feeder"],
        textfont=dict(color=text_primary, size=12),
        textposition="top center",
        hovertext=map_df["Type"] + " (" + map_df["Capacity"] + ") - " + map_df["Status"],
        hoverinfo="text"
    ))

    active_coord = KAIMUR_COORDS[selected_dt]
    map_fig.update_layout(
        map=dict(
            style=map_style,
            center=dict(lat=active_coord["lat"], lon=active_coord["lon"]),
            zoom=10.2
        ),
        height=230,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=card_bg
    )
    st.plotly_chart(map_fig, use_container_width=True)

    # Clean Health & Outage Diagnosis Box
    if pred_health_tier == 0:
        st.success(
            f"**Status: {status_text} ({health_score}/100)** — Normal operating condition (`{pred_cause_label}`). "
            f"Winding temperature at **{pred_hotspot_c:.1f}°C** is within safe limits."
        )
    elif pred_health_tier == 1:
        st.warning(
            f"**Status: {status_text} ({health_score}/100)** — High load or thermal stress (**{overload_ratio*100:.0f}%** load, **{pred_hotspot_c:.1f}°C** winding temp). "
            f"Monitor for `{pred_cause_label}` and check cooling."
        )
    else:
        st.error(
            f"**Status: {status_text} ({health_score}/100)** — High risk of **{pred_cause_label}**. "
            f"Estimated **{pred_blackout_hours:.1f}-hour blackout** at {active_coord['substation']} unless load is reduced."
        )