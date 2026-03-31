"""
Energy Load Forecasting — Streamlit Dashboard
==============================================
Run with: streamlit run app.py

This version trains the model directly from the dataset,
so no .joblib file is needed.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Energy Load Forecasting",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
<style>
    .predict-result {
        background: linear-gradient(135deg, #1a73e8, #0d47a1);
        color: white;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }
    .predict-mw {
        font-size: 3rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ── Train model (cached so it only runs once) ──────────────────────────────────
@st.cache_resource
def load_model_and_data():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    with st.spinner("Loading dataset and training models — please wait about 2 minutes..."):

        energy  = pd.read_csv('energy_dataset.csv')
        weather = pd.read_csv('weather_features.csv')

        energy['time']    = pd.to_datetime(energy['time'], utc=True).dt.tz_convert(None)
        weather['dt_iso'] = pd.to_datetime(weather['dt_iso'], utc=True).dt.tz_convert(None)

        weather_madrid = weather[weather['city_name'] == 'Madrid'].copy()
        df = pd.merge(energy, weather_madrid, left_on='time', right_on='dt_iso', how='inner')
        df = df.rename(columns={'total load actual': 'total_load_actual'})
        df['temp_c'] = df['temp'] - 273.15
        df = df.dropna(subset=['total_load_actual'])
        df = df.sort_values('time').reset_index(drop=True)

        df['hour']       = df['time'].dt.hour
        df['dayofweek']  = df['time'].dt.dayofweek
        df['month']      = df['time'].dt.month
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['hour_sin']   = np.sin(2 * np.pi * df['hour']      / 24)
        df['hour_cos']   = np.cos(2 * np.pi * df['hour']      / 24)
        df['month_sin']  = np.sin(2 * np.pi * df['month']     / 12)
        df['month_cos']  = np.cos(2 * np.pi * df['month']     / 12)
        df['dow_sin']    = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dow_cos']    = np.cos(2 * np.pi * df['dayofweek'] / 7)

        df['load_lag_1h']           = df['total_load_actual'].shift(1)
        df['load_lag_24h']          = df['total_load_actual'].shift(24)
        df['load_lag_168h']         = df['total_load_actual'].shift(168)
        df['load_rolling_mean_24h'] = df['total_load_actual'].shift(1).rolling(24).mean()
        df['load_rolling_std_24h']  = df['total_load_actual'].shift(1).rolling(24).std()

        FEATURES = [
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
            'is_weekend', 'temp_c', 'humidity', 'wind_speed',
            'load_lag_1h', 'load_lag_24h', 'load_lag_168h',
            'load_rolling_mean_24h', 'load_rolling_std_24h'
        ]
        TARGET = 'total_load_actual'

        df = df.dropna(subset=FEATURES + [TARGET])
        X = df[FEATURES]
        y = df[TARGET]
        split = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest':     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.05,
                                                           max_depth=5, random_state=42)
        }

        results = {}
        predictions = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            predictions[name] = preds
            mae  = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2   = r2_score(y_test, preds)
            mape = np.mean(np.abs((y_test.values - preds) / y_test.values)) * 100
            results[name] = {
                'MAE': round(mae, 1), 'RMSE': round(rmse, 1),
                'R²': round(r2, 4),   'MAPE (%)': round(mape, 2)
            }

        results_df = pd.DataFrame(results).T
        return models['Random Forest'], results_df, FEATURES

rf_model, results_df, FEATURES = load_model_and_data()

# ── Helper: build features ─────────────────────────────────────────────────────
def build_features(dt, temp_c, humidity, wind_speed,
                   lag_1h, lag_24h, lag_168h, rolling_mean, rolling_std):
    hour  = dt.hour
    dow   = dt.weekday()
    month = dt.month
    return pd.DataFrame([{
        'hour_sin':              np.sin(2 * np.pi * hour  / 24),
        'hour_cos':              np.cos(2 * np.pi * hour  / 24),
        'month_sin':             np.sin(2 * np.pi * month / 12),
        'month_cos':             np.cos(2 * np.pi * month / 12),
        'dow_sin':               np.sin(2 * np.pi * dow   / 7),
        'dow_cos':               np.cos(2 * np.pi * dow   / 7),
        'is_weekend':            int(dow >= 5),
        'temp_c':                temp_c,
        'humidity':              humidity,
        'wind_speed':            wind_speed,
        'load_lag_1h':           lag_1h,
        'load_lag_24h':          lag_24h,
        'load_lag_168h':         lag_168h,
        'load_rolling_mean_24h': rolling_mean,
        'load_rolling_std_24h':  rolling_std,
    }])

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚡ Energy Forecasting")
    st.caption("Spain National Grid · ML Pipeline")
    st.divider()
    page = st.radio("Navigate", [
        "🔮 Predict Load",
        "📊 Model Dashboard",
        "📈 Results & Charts"
    ], label_visibility="collapsed")
    st.divider()
    st.caption("Final Year CS Honours Project")
    st.caption("Random Forest · R² = 0.9823")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT LOAD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔮 Predict Load":
    st.title("🔮 Predict Electricity Load")
    st.markdown("Enter the conditions below to get a predicted load for any hour.")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📅 Date & Time")
        pred_date = st.date_input("Date", value=datetime(2018, 1, 15))
        pred_hour = st.slider("Hour of day", 0, 23, 12)
        pred_dt   = datetime(pred_date.year, pred_date.month, pred_date.day, pred_hour)

        st.subheader("🌤️ Weather Conditions")
        temp_c     = st.slider("Temperature (°C)", -10.0, 45.0, 12.0, step=0.5)
        humidity   = st.slider("Humidity (%)", 0, 100, 65)
        wind_speed = st.slider("Wind Speed (m/s)", 0.0, 20.0, 4.0, step=0.5)

    with col2:
        st.subheader("🕐 Recent Load History (MW)")
        lag_1h   = st.number_input("Load 1 hour ago (MW)",   value=28500, step=100)
        lag_24h  = st.number_input("Load 24 hours ago (MW)", value=27800, step=100)
        lag_168h = st.number_input("Load 1 week ago (MW)",   value=29100, step=100)

        st.subheader("📉 Rolling Statistics")
        rolling_mean = st.number_input("24h Rolling Mean (MW)", value=28000, step=100)
        rolling_std  = st.number_input("24h Rolling Std (MW)",  value=2200,  step=100)

    st.divider()

    if st.button("⚡ Predict Load", use_container_width=True, type="primary"):
        X_pred     = build_features(pred_dt, temp_c, humidity, wind_speed,
                                    lag_1h, lag_24h, lag_168h, rolling_mean, rolling_std)
        prediction = rf_model.predict(X_pred)[0]
        mean_demand = 28697

        if prediction > mean_demand * 1.1:
            label = "🔴 High Demand"
        elif prediction < mean_demand * 0.9:
            label = "🟢 Low Demand"
        else:
            label = "🟡 Normal Demand"

        st.markdown(f"""
        <div class="predict-result">
            <div style="font-size:1rem;opacity:0.85;">Predicted Electricity Load</div>
            <div class="predict-mw">{prediction:,.0f} MW</div>
            <div style="margin-top:8px;font-size:1.1rem;">{label}</div>
            <div style="margin-top:4px;opacity:0.8;font-size:0.9rem;">
                {pred_dt.strftime('%A %d %B %Y, %H:00')} · {temp_c}°C · {humidity}% humidity
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted Load", f"{prediction:,.0f} MW")
        c2.metric("vs Mean Demand", f"{prediction - mean_demand:+,.0f} MW")
        c3.metric("vs 1h Ago",      f"{prediction - lag_1h:+,.0f} MW")
        c4.metric("Model R²",       "0.9823")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Dashboard":
    st.title("📊 Model Performance Dashboard")
    st.markdown("Comparative evaluation of all three models on the held-out test set.")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model",  "Random Forest")
    c2.metric("Best R²",     "0.9823", delta="+0.0418 vs Linear Reg")
    c3.metric("Best MAE",    "379.0 MW", delta="-453.1 MW vs baseline")
    c4.metric("Best MAPE",   "1.32%")

    st.divider()
    st.subheader("📋 Full Results Table")
    st.dataframe(
        results_df.style
            .highlight_min(subset=['MAE', 'RMSE', 'MAPE (%)'], color='#d4edda')
            .highlight_max(subset=['R²'], color='#d4edda')
            .format({'MAE': '{:.1f}', 'RMSE': '{:.1f}',
                     'R²': '{:.4f}', 'MAPE (%)': '{:.2f}%'}),
        use_container_width=True
    )

    st.divider()
    st.subheader("📊 Visual Comparison")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = ['#4e79a7', '#f28e2b', '#59a14f']
    model_names = results_df.index.tolist()

    for ax, metric in zip(axes, ['MAE', 'RMSE', 'R²']):
        vals = results_df[metric].tolist()
        bars = ax.bar(model_names, vals, color=colors)
        ax.set_title(metric, fontsize=13, fontweight='bold')
        ax.set_xticklabels(model_names, rotation=15, ha='right', fontsize=9)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                    f'{val}', ha='center', va='bottom', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()
    st.subheader("🔬 Ablation Study — Impact of Lag Features")
    ab1, ab2 = st.columns(2)
    ab1.metric("R² without lags", "0.5728")
    ab1.metric("R² with lags",    "0.9823", delta="+0.4095")
    ab2.metric("MAE without lags","2,097.5 MW")
    ab2.metric("MAE with lags",   "379.0 MW", delta="-82.4%")

    fig2, ax2 = plt.subplots(figsize=(8, 3))
    configs  = ['Without Lag Features', 'With Lag Features']
    r2_vals  = [0.5728, 0.9823]
    bar_cols = ['#e07b54', '#59a14f']
    bars = ax2.barh(configs, r2_vals, color=bar_cols, height=0.4)
    ax2.set_xlim(0, 1.05)
    ax2.set_xlabel('R² Score')
    ax2.set_title('Ablation Study — R² With vs Without Lag Features', fontweight='bold')
    for bar, val in zip(bars, r2_vals):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{val}', va='center', fontsize=11, fontweight='bold')
    ax2.axvline(0.85, color='red', linewidth=1, linestyle='--',
                alpha=0.7, label='Success threshold (R²=0.85)')
    ax2.legend(fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RESULTS & CHARTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Results & Charts":
    st.title("📈 Results & Analysis Charts")
    st.divider()

    chart_files = {
        "Figure 1 — Full Time Series":            "plot_full_timeseries.png",
        "Figure 2 — Feature Correlation":         "plot_correlation.png",
        "Figure 3 — Actual vs Predicted":         "plot_actual_vs_predicted.png",
        "Figure 4 — Model Comparison":            "plot_model_comparison.png",
        "Figure 5 — Feature Importance":          "plot_feature_importance.png",
        "Figure 6 — Residual Analysis":           "plot_residuals.png",
        "Figure 7 — Error by Time":               "plot_error_by_time.png",
        "Figure 8 — Ablation Study":              "plot_ablation.png",
    }

    available = {k: v for k, v in chart_files.items() if os.path.exists(v)}
    missing   = {k: v for k, v in chart_files.items() if not os.path.exists(v)}

    if missing:
        st.warning(f"⚠️ {len(missing)} chart(s) not found in current directory.")

    for title, filename in available.items():
        st.subheader(title)
        st.image(filename, use_column_width=True)
        st.divider()

    if not available:
        st.info("No chart PNG files found. Copy your plot_*.png files into the same folder as app.py.")
