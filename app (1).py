"""
Energy Load Forecasting — Streamlit Dashboard
==============================================
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
        color: white; border-radius: 12px;
        padding: 24px; text-align: center;
    }
    .predict-mw { font-size: 3rem; font-weight: bold; }
    .about-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1a73e8;
        border-radius: 4px;
        padding: 12px 16px;
        font-size: 0.85rem; color: #444;
    }
    .template-box {
        background-color: #e8f5e9;
        border-left: 4px solid #59a14f;
        border-radius: 4px;
        padding: 12px 16px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Train model (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_data():
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    with st.spinner("Training models — please wait about 2 minutes on first load..."):

        energy  = pd.read_csv('energy_dataset.csv')
        weather = pd.read_csv('weather_features.csv')

        energy['time']    = pd.to_datetime(energy['time'], utc=True).dt.tz_convert(None)
        weather['dt_iso'] = pd.to_datetime(weather['dt_iso'], utc=True).dt.tz_convert(None)

        wm = weather[weather['city_name'] == 'Madrid'].copy()
        df = pd.merge(energy, wm, left_on='time', right_on='dt_iso', how='inner')
        df = df.rename(columns={'total load actual': 'total_load_actual'})
        df['temp_c'] = df['temp'] - 273.15
        df = df.dropna(subset=['total_load_actual'])
        df = df.sort_values('time').reset_index(drop=True)

        df = _add_features(df)

        FEATURES = [
            'hour_sin','hour_cos','month_sin','month_cos','dow_sin','dow_cos',
            'is_weekend','temp_c','humidity','wind_speed',
            'load_lag_1h','load_lag_24h','load_lag_168h',
            'load_rolling_mean_24h','load_rolling_std_24h'
        ]
        TARGET = 'total_load_actual'
        df = df.dropna(subset=FEATURES + [TARGET])

        X = df[FEATURES]; y = df[TARGET]
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
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mae  = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2   = r2_score(y_test, preds)
            mape = np.mean(np.abs((y_test.values - preds) / y_test.values)) * 100
            results[name] = {'MAE': round(mae,1), 'RMSE': round(rmse,1),
                             'R²': round(r2,4),   'MAPE (%)': round(mape,2)}

        return models['Random Forest'], pd.DataFrame(results).T, FEATURES, df

def _add_features(df):
    df['hour']      = df['time'].dt.hour
    df['dayofweek'] = df['time'].dt.dayofweek
    df['month']     = df['time'].dt.month
    df['is_weekend']= (df['dayofweek'] >= 5).astype(int)
    df['hour_sin']  = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos']  = np.cos(2*np.pi*df['hour']/24)
    df['month_sin'] = np.sin(2*np.pi*df['month']/12)
    df['month_cos'] = np.cos(2*np.pi*df['month']/12)
    df['dow_sin']   = np.sin(2*np.pi*df['dayofweek']/7)
    df['dow_cos']   = np.cos(2*np.pi*df['dayofweek']/7)
    if 'total_load_actual' in df.columns:
        df['load_lag_1h']           = df['total_load_actual'].shift(1)
        df['load_lag_24h']          = df['total_load_actual'].shift(24)
        df['load_lag_168h']         = df['total_load_actual'].shift(168)
        df['load_rolling_mean_24h'] = df['total_load_actual'].shift(1).rolling(24).mean()
        df['load_rolling_std_24h']  = df['total_load_actual'].shift(1).rolling(24).std()
    return df

rf_model, results_df, FEATURES, full_df = load_model_and_data()

# ── Single prediction helper ───────────────────────────────────────────────────
def build_single(dt, temp_c, humidity, wind_speed,
                 lag_1h, lag_24h, lag_168h, rmean, rstd):
    h, d, m = dt.hour, dt.weekday(), dt.month
    return pd.DataFrame([{
        'hour_sin': np.sin(2*np.pi*h/24), 'hour_cos': np.cos(2*np.pi*h/24),
        'month_sin': np.sin(2*np.pi*m/12), 'month_cos': np.cos(2*np.pi*m/12),
        'dow_sin': np.sin(2*np.pi*d/7),   'dow_cos': np.cos(2*np.pi*d/7),
        'is_weekend': int(d>=5), 'temp_c': temp_c,
        'humidity': humidity, 'wind_speed': wind_speed,
        'load_lag_1h': lag_1h, 'load_lag_24h': lag_24h, 'load_lag_168h': lag_168h,
        'load_rolling_mean_24h': rmean, 'load_rolling_std_24h': rstd,
    }])

# ── Results chart helper ───────────────────────────────────────────────────────
def show_results_charts(df_r, title=""):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    has_actual = 'total_load_actual' in df_r.columns and df_r['total_load_actual'].notna().any()

    if has_actual:
        mae  = mean_absolute_error(df_r['total_load_actual'], df_r['predicted'])
        rmse = np.sqrt(mean_squared_error(df_r['total_load_actual'], df_r['predicted']))
        r2   = r2_score(df_r['total_load_actual'], df_r['predicted'])
        mape = (((df_r['total_load_actual'] - df_r['predicted']).abs() /
                  df_r['total_load_actual']).mean() * 100)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Hours", f"{len(df_r):,}")
        c2.metric("MAE",  f"{mae:.0f} MW")
        c3.metric("R²",   f"{r2:.4f}")
        c4.metric("MAPE", f"{mape:.2f}%")
        st.divider()

        # Actual vs Predicted
        st.subheader("📈 Actual vs Predicted")
        fig, ax = plt.subplots(figsize=(14,4))
        ax.plot(df_r['time'], df_r['total_load_actual'], label='Actual',
                color='black', linewidth=1.2)
        ax.plot(df_r['time'], df_r['predicted'],
                label=f'Predicted (R²={r2:.4f})', color='steelblue',
                linewidth=1.2, alpha=0.85)
        ax.fill_between(df_r['time'],
                        df_r['predicted']-mae, df_r['predicted']+mae,
                        alpha=0.15, color='steelblue', label=f'±MAE ({mae:.0f} MW)')
        ax.set_ylabel('Load (MW)'); ax.set_xlabel('Date')
        ax.set_title(f'Actual vs Predicted — {title}', fontsize=13)
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.xticks(rotation=30); plt.tight_layout()
        st.pyplot(fig); plt.close()

        # Error bar chart
        st.subheader("📉 Prediction Error Over Time")
        df_r['error'] = df_r['predicted'] - df_r['total_load_actual']
        fig2, ax2 = plt.subplots(figsize=(14,3))
        colors = df_r['error'].apply(lambda x: '#e07b54' if x>0 else '#59a14f')
        ax2.bar(df_r['time'], df_r['error'], color=colors, width=0.03, alpha=0.7)
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.axhline(mae, color='red', linewidth=1, linestyle='--', alpha=0.5)
        ax2.axhline(-mae, color='red', linewidth=1, linestyle='--', alpha=0.5,
                    label=f'±MAE ({mae:.0f} MW)')
        ax2.set_ylabel('Error (MW)'); ax2.set_xlabel('Date')
        ax2.set_title('Prediction Error (Predicted − Actual)', fontsize=12)
        ax2.legend(fontsize=9)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
        plt.xticks(rotation=30); plt.tight_layout()
        st.pyplot(fig2); plt.close()

        # MAE by hour
        st.subheader("🕐 MAE by Hour of Day")
        df_r['abs_error'] = df_r['error'].abs()
        err_hour = df_r.groupby('hour')['abs_error'].mean()
        fig3, ax3 = plt.subplots(figsize=(12,3))
        ax3.bar(err_hour.index, err_hour.values, color='steelblue')
        ax3.set_xlabel('Hour of Day'); ax3.set_ylabel('MAE (MW)')
        ax3.set_title('Mean Absolute Error by Hour of Day')
        ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)
        plt.tight_layout(); st.pyplot(fig3); plt.close()

    else:
        st.info("No actual load values in file — showing predictions only.")
        fig, ax = plt.subplots(figsize=(14,4))
        ax.plot(df_r['time'], df_r['predicted'], color='steelblue', linewidth=1.2)
        ax.set_ylabel('Predicted Load (MW)'); ax.set_xlabel('Date')
        ax.set_title('Predicted Load', fontsize=13)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.xticks(rotation=30); plt.tight_layout()
        st.pyplot(fig); plt.close()

    # Download
    st.divider()
    st.subheader("⬇️ Download Predictions")
    cols = ['time','predicted']
    if has_actual:
        cols += ['total_load_actual','error','abs_error']
    dl = df_r[cols].copy()
    dl.columns = (['Datetime','Predicted Load (MW)'] +
                  (['Actual Load (MW)','Error (MW)','Absolute Error (MW)'] if has_actual else []))
    st.download_button("📥 Download as CSV", dl.to_csv(index=False),
                       file_name="predictions.csv", mime="text/csv",
                       use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚡ Energy Forecasting")
    st.caption("Spain National Grid · ML Pipeline")
    st.divider()
    page = st.radio("Navigate", [
        "🔮 Single Hour Prediction",
        "📅 Date Range Forecast",
        "📤 Upload Your Own Data",
        "🌍 Live European Grid",
        "📊 Model Dashboard",
        "📈 Results & Charts"
    ], label_visibility="collapsed")
    st.divider()
    st.markdown("""
    <div class="about-box">
    <b>About this app</b><br><br>
    Final Year CS Honours Project · LSBU<br><br>
    Random Forest model trained on 36,063 hourly observations from the Spanish national grid (2015–2018).<br><br>
    <b>R² = 0.9823 · MAE = 379 MW</b>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SINGLE HOUR PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔮 Single Hour Prediction":
    st.title("🔮 Predict Electricity Load")
    st.markdown("Enter conditions below to predict electricity demand for a specific hour.")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📅 Date & Time")
        pred_date = st.date_input("Date", value=datetime(2018,1,15))
        pred_hour = st.slider("Hour of day", 0, 23, 12)
        pred_dt   = datetime(pred_date.year, pred_date.month, pred_date.day, pred_hour)

        st.subheader("🌤️ Weather Conditions")
        temp_c     = st.slider("Temperature (°C)", -10.0, 45.0, 12.0, step=0.5)
        humidity   = st.slider("Humidity (%)", 0, 100, 65)
        wind_speed = st.slider("Wind Speed (m/s)", 0.0, 20.0, 4.0, step=0.5)

    with col2:
        st.subheader("🕐 Recent Load History (MW)")
        st.info("💡 These are recent electricity demand readings. Leave defaults if unsure — they represent typical Spanish grid demand.")
        lag_1h   = st.number_input("Load 1 hour ago (MW)",   value=28500, step=100,
                                    help="Electricity demand 1 hour before prediction time")
        lag_24h  = st.number_input("Load 24 hours ago (MW)", value=27800, step=100,
                                    help="Same hour yesterday")
        lag_168h = st.number_input("Load 1 week ago (MW)",   value=29100, step=100,
                                    help="Same hour last week")
        st.subheader("📉 Rolling Statistics")
        rmean = st.number_input("24h Rolling Mean (MW)", value=28000, step=100,
                                 help="Average demand over past 24 hours")
        rstd  = st.number_input("24h Rolling Std (MW)",  value=2200,  step=100,
                                 help="Variability of demand over past 24 hours")

    st.divider()
    if st.button("⚡ Predict Load", use_container_width=True, type="primary"):
        pred = rf_model.predict(build_single(pred_dt,temp_c,humidity,wind_speed,
                                             lag_1h,lag_24h,lag_168h,rmean,rstd))[0]
        mean_d = 28697
        label  = "🔴 High Demand" if pred>mean_d*1.1 else ("🟢 Low Demand" if pred<mean_d*0.9 else "🟡 Normal Demand")
        st.markdown(f"""
        <div class="predict-result">
            <div style="font-size:1rem;opacity:0.85;">Predicted Electricity Load</div>
            <div class="predict-mw">{pred:,.0f} MW</div>
            <div style="margin-top:8px;font-size:1.1rem;">{label}</div>
            <div style="margin-top:4px;opacity:0.8;font-size:0.9rem;">
                {pred_dt.strftime('%A %d %B %Y, %H:00')} · {temp_c}°C · {humidity}% humidity
            </div>
        </div>""", unsafe_allow_html=True)
        st.divider()
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Predicted Load", f"{pred:,.0f} MW")
        c2.metric("vs Mean Demand", f"{pred-mean_d:+,.0f} MW")
        c3.metric("vs 1h Ago",      f"{pred-lag_1h:+,.0f} MW")
        c4.metric("Model R²",       "0.9823")
        st.caption("ℹ️ Model MAE = 379 MW — actual demand is typically within ±379 MW of this prediction.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATE RANGE FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📅 Date Range Forecast":
    st.title("📅 Date Range Forecast")
    st.markdown("Select a date range within the dataset (2015–2018) to see predicted vs actual demand.")
    st.divider()

    min_d = full_df['time'].min().date()
    max_d = full_df['time'].max().date()
    c1,c2 = st.columns(2)
    with c1:
        start = st.date_input("Start date", value=datetime(2018,1,1).date(),
                               min_value=min_d, max_value=max_d)
    with c2:
        end   = st.date_input("End date",   value=datetime(2018,1,7).date(),
                               min_value=min_d, max_value=max_d)

    if start >= end:
        st.error("End date must be after start date.")
    else:
        if (end-start).days > 90:
            st.warning("⚠️ Large ranges may be slow. Try keeping under 90 days.")
        mask   = (full_df['time'].dt.date >= start) & (full_df['time'].dt.date <= end)
        rdf    = full_df[mask].copy()
        if len(rdf) == 0:
            st.error("No data for this range.")
        else:
            rdf['predicted'] = rf_model.predict(rdf[FEATURES])
            show_results_charts(rdf, f"{start} to {end}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — UPLOAD YOUR OWN DATA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📤 Upload Your Own Data":
    st.title("📤 Upload Your Own Data")
    st.markdown("Upload a CSV file and the model will generate predictions and charts automatically.")
    st.divider()

    fmt = st.radio("Choose your file format", [
        "📋 Simple format — I have time, weather and load values",
        "🔬 Advanced format — I have the full energy + weather dataset format"
    ])

    # ── Template downloads ────────────────────────────────────────────────────
    st.subheader("📥 Download a Template")

    if "Simple" in fmt:
        st.markdown("""
        <div class="template-box">
        <b>Simple format columns required:</b><br>
        <code>time, temp_c, humidity, wind_speed, total_load_actual</code><br><br>
        • <b>time</b> — datetime e.g. 2018-06-01 14:00:00<br>
        • <b>temp_c</b> — temperature in Celsius<br>
        • <b>humidity</b> — humidity percentage (0–100)<br>
        • <b>wind_speed</b> — wind speed in m/s<br>
        • <b>total_load_actual</b> — actual electricity load in MW (optional — used for comparison)
        </div>
        """, unsafe_allow_html=True)

        # Generate simple template
        template_simple = pd.DataFrame({
            'time':               ['2018-06-01 00:00:00', '2018-06-01 01:00:00',
                                   '2018-06-01 02:00:00', '2018-06-01 03:00:00'],
            'temp_c':             [18.5, 17.8, 17.2, 16.9],
            'humidity':           [65, 67, 68, 70],
            'wind_speed':         [3.2, 2.8, 3.1, 2.5],
            'total_load_actual':  [24500, 23200, 22800, 22400],
        })
        st.download_button("📥 Download Simple Template",
                           template_simple.to_csv(index=False),
                           file_name="template_simple.csv",
                           mime="text/csv")
    else:
        st.markdown("""
        <div class="template-box">
        <b>Advanced format:</b> Same structure as the original <code>energy_dataset.csv</code> and
        <code>weather_features.csv</code> files from Kaggle.<br><br>
        The app will merge them automatically — just upload both files below.
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── File upload ───────────────────────────────────────────────────────────
    st.subheader("📂 Upload Your File(s)")

    if "Simple" in fmt:
        uploaded = st.file_uploader("Upload your CSV file", type=['csv'])

        if uploaded:
            try:
                udf = pd.read_csv(uploaded)
                udf['time'] = pd.to_datetime(udf['time'])
                udf = udf.sort_values('time').reset_index(drop=True)

                # Validate columns
                required = ['time','temp_c','humidity','wind_speed']
                missing_cols = [c for c in required if c not in udf.columns]
                if missing_cols:
                    st.error(f"Missing columns: {missing_cols}. Please check your file matches the template.")
                else:
                    st.success(f"✅ File loaded — {len(udf):,} rows, {udf['time'].min().date()} to {udf['time'].max().date()}")

                    # Add temporal features
                    udf['hour']      = udf['time'].dt.hour
                    udf['dayofweek'] = udf['time'].dt.dayofweek
                    udf['month']     = udf['time'].dt.month
                    udf['is_weekend']= (udf['dayofweek'] >= 5).astype(int)
                    udf['hour_sin']  = np.sin(2*np.pi*udf['hour']/24)
                    udf['hour_cos']  = np.cos(2*np.pi*udf['hour']/24)
                    udf['month_sin'] = np.sin(2*np.pi*udf['month']/12)
                    udf['month_cos'] = np.cos(2*np.pi*udf['month']/12)
                    udf['dow_sin']   = np.sin(2*np.pi*udf['dayofweek']/7)
                    udf['dow_cos']   = np.cos(2*np.pi*udf['dayofweek']/7)

                    # Lag features (from actual load if available, else mean)
                    if 'total_load_actual' in udf.columns:
                        udf['load_lag_1h']           = udf['total_load_actual'].shift(1)
                        udf['load_lag_24h']           = udf['total_load_actual'].shift(24)
                        udf['load_lag_168h']          = udf['total_load_actual'].shift(168)
                        udf['load_rolling_mean_24h']  = udf['total_load_actual'].shift(1).rolling(24).mean()
                        udf['load_rolling_std_24h']   = udf['total_load_actual'].shift(1).rolling(24).std()
                    else:
                        mean_load = 28697
                        udf['load_lag_1h']           = mean_load
                        udf['load_lag_24h']           = mean_load
                        udf['load_lag_168h']          = mean_load
                        udf['load_rolling_mean_24h']  = mean_load
                        udf['load_rolling_std_24h']   = 2200

                    udf = udf.dropna(subset=FEATURES)

                    if len(udf) == 0:
                        st.warning("Not enough rows to generate predictions — need at least 168 rows for lag features.")
                    else:
                        udf['predicted'] = rf_model.predict(udf[FEATURES])
                        st.divider()
                        show_results_charts(udf, "Uploaded Data")

            except Exception as e:
                st.error(f"Error reading file: {e}. Please check your file matches the template format.")

    else:
        # Advanced — two file upload
        col1, col2 = st.columns(2)
        with col1:
            energy_file  = st.file_uploader("Upload energy_dataset.csv", type=['csv'])
        with col2:
            weather_file = st.file_uploader("Upload weather_features.csv", type=['csv'])

        if energy_file and weather_file:
            try:
                energy_u  = pd.read_csv(energy_file)
                weather_u = pd.read_csv(weather_file)

                energy_u['time']    = pd.to_datetime(energy_u['time'], utc=True).dt.tz_convert(None)
                weather_u['dt_iso'] = pd.to_datetime(weather_u['dt_iso'], utc=True).dt.tz_convert(None)

                wm_u = weather_u[weather_u['city_name'] == 'Madrid'].copy()
                udf  = pd.merge(energy_u, wm_u, left_on='time', right_on='dt_iso', how='inner')
                udf  = udf.rename(columns={'total load actual': 'total_load_actual'})
                udf['temp_c'] = udf['temp'] - 273.15
                udf  = udf.dropna(subset=['total_load_actual'])
                udf  = udf.sort_values('time').reset_index(drop=True)
                udf  = _add_features(udf)
                udf  = udf.dropna(subset=FEATURES)

                st.success(f"✅ Files merged — {len(udf):,} rows")
                udf['predicted'] = rf_model.predict(udf[FEATURES])
                st.divider()
                show_results_charts(udf, "Uploaded Dataset")

            except Exception as e:
                st.error(f"Error processing files: {e}")
        else:
            st.info("Please upload both files to continue.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Dashboard":
    st.title("📊 Model Performance Dashboard")
    st.markdown("Comparative evaluation of all three models on the held-out test set (7,213 hours).")
    st.divider()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Best Model", "Random Forest")
    c2.metric("Best R²",    "0.9823", delta="+0.0418 vs Linear Reg")
    c3.metric("Best MAE",   "379.0 MW", delta="-453.1 MW vs baseline")
    c4.metric("Best MAPE",  "1.32%")
    st.divider()

    st.subheader("📋 Full Results Table")
    st.dataframe(
        results_df.style
            .highlight_min(subset=['MAE','RMSE','MAPE (%)'], color='#d4edda')
            .highlight_max(subset=['R²'], color='#d4edda')
            .format({'MAE':'{:.1f}','RMSE':'{:.1f}','R²':'{:.4f}','MAPE (%)':'{:.2f}%'}),
        use_container_width=True
    )
    st.divider()

    st.subheader("📊 Visual Comparison")
    fig, axes = plt.subplots(1, 3, figsize=(14,4))
    colors = ['#4e79a7','#f28e2b','#59a14f']
    for ax, metric in zip(axes, ['MAE','RMSE','R²']):
        vals = results_df[metric].tolist()
        bars = ax.bar(results_df.index.tolist(), vals, color=colors)
        ax.set_title(metric, fontsize=13, fontweight='bold')
        ax.set_xticklabels(results_df.index.tolist(), rotation=15, ha='right', fontsize=9)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                    f'{val}', ha='center', va='bottom', fontsize=9)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.divider()
    st.subheader("🔬 Ablation Study — Impact of Lag Features")
    ab1,ab2 = st.columns(2)
    ab1.metric("R² without lags","0.5728")
    ab1.metric("R² with lags",   "0.9823", delta="+0.4095")
    ab2.metric("MAE without lags","2,097.5 MW")
    ab2.metric("MAE with lags",   "379.0 MW", delta="-82.4%")

    fig2, ax2 = plt.subplots(figsize=(8,3))
    bars = ax2.barh(['Without Lags','With Lags'], [0.5728,0.9823],
                    color=['#e07b54','#59a14f'], height=0.4)
    ax2.set_xlim(0,1.05); ax2.set_xlabel('R² Score')
    ax2.set_title('Ablation Study — R² With vs Without Lag Features', fontweight='bold')
    for bar, val in zip(bars, [0.5728,0.9823]):
        ax2.text(val+0.01, bar.get_y()+bar.get_height()/2,
                 f'{val}', va='center', fontsize=11, fontweight='bold')
    ax2.axvline(0.85, color='red', linewidth=1, linestyle='--', alpha=0.7,
                label='Success threshold (R²=0.85)')
    ax2.legend(fontsize=9)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    plt.tight_layout(); st.pyplot(fig2); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — RESULTS & CHARTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Results & Charts":
    st.title("📈 Results & Analysis Charts")
    st.markdown("All charts generated from the project notebook pipeline.")
    st.divider()

    chart_files = {
        "Figure 1 — Full Time Series":    "plot_full_timeseries.png",
        "Figure 2 — Feature Correlation": "plot_correlation.png",
        "Figure 3 — Actual vs Predicted": "plot_actual_vs_predicted.png",
        "Figure 4 — Model Comparison":    "plot_model_comparison.png",
        "Figure 5 — Feature Importance":  "plot_feature_importance.png",
        "Figure 6 — Residual Analysis":   "plot_residuals.png",
        "Figure 7 — Error by Time":       "plot_error_by_time.png",
        "Figure 8 — Ablation Study":      "plot_ablation.png",
    }
    available = {k:v for k,v in chart_files.items() if os.path.exists(v)}
    missing   = {k:v for k,v in chart_files.items() if not os.path.exists(v)}

    if missing:
        st.warning(f"⚠️ {len(missing)} chart(s) not found. Make sure all plot_*.png files are uploaded to GitHub.")
    for title, filename in available.items():
        st.subheader(title)
        st.image(filename, use_column_width=True)
        st.divider()
    if not available:
        st.info("No chart PNG files found.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — LIVE EUROPEAN GRID DATA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌍 Live European Grid":
    import requests
    from xml.etree import ElementTree as ET

    st.title("🌍 Live European Grid Forecast")
    st.markdown("Fetch real electricity demand and weather data for any European country and run the model on it.")
    st.divider()

    # Country options with ENTSO-E domain codes
    COUNTRIES = {
        "Portugal": "10YPT-REN------W",
        "Spain":    "10YES-REE------0",
        "France":   "10YFR-RTE------C",
        "Germany":  "10Y1001A1001A83F",
        "Italy":    "10YIT-GRTN-----B",
        "Netherlands": "10YNL----------L",
        "Belgium":  "10YBE----------2",
        "Poland":   "10YPL-AREA-----S",
    }

    CITY_COORDS = {
        "Portugal":    (38.7, -9.1),
        "Spain":       (40.4, -3.7),
        "France":      (48.9, 2.3),
        "Germany":     (52.5, 13.4),
        "Italy":       (41.9, 12.5),
        "Netherlands": (52.4, 4.9),
        "Belgium":     (50.8, 4.4),
        "Poland":      (52.2, 21.0),
    }

    ENTSO_API_KEY = "335b12c9-dea9-483f-8bb1-62896a6def14"

    col1, col2 = st.columns(2)
    with col1:
        country = st.selectbox("Select Country", list(COUNTRIES.keys()))
    with col2:
        n_days = st.slider("Number of days to fetch", 1, 14, 3)

    from datetime import timedelta
    end_dt   = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=n_days)

    if st.button("🔄 Fetch Live Data & Predict", use_container_width=True, type="primary"):

        domain  = COUNTRIES[country]
        lat, lon = CITY_COORDS[country]

        # ── Step 1: Fetch load from ENTSO-E ──────────────────────────────
        with st.spinner(f"Fetching {country} grid data from ENTSO-E..."):
            try:
                period_start = start_dt.strftime('%Y%m%d%H%M')
                period_end   = end_dt.strftime('%Y%m%d%H%M')

                url = (
                    f"https://web-api.tp.entsoe.eu/api"
                    f"?documentType=A65"
                    f"&processType=A16"
                    f"&outBiddingZone_Domain={domain}"
                    f"&periodStart={period_start}"
                    f"&periodEnd={period_end}"
                    f"&securityToken={ENTSO_API_KEY}"
                )

                resp = requests.get(url, timeout=30)

                if resp.status_code == 200:
                    root = ET.fromstring(resp.content)
                    ns   = {'ns': 'urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0'}

                    times, loads = [], []
                    for ts in root.findall('.//ns:TimeSeries', ns):
                        period = ts.find('ns:Period', ns)
                        if period is None:
                            continue
                        start_str = period.find('ns:timeInterval/ns:start', ns).text
                        start_ts  = pd.to_datetime(start_str)
                        for pt in period.findall('ns:Point', ns):
                            pos  = int(pt.find('ns:position', ns).text)
                            qty  = pt.find('ns:quantity', ns)
                            if qty is not None:
                                times.append(start_ts + pd.Timedelta(hours=pos-1))
                                loads.append(float(qty.text))

                    if not times:
                        st.error("No load data returned. The ENTSO-E API may not have data for this period yet.")
                        st.stop()

                    load_df = pd.DataFrame({'time': times, 'total_load_actual': loads})
                    load_df = load_df.sort_values('time').drop_duplicates('time').reset_index(drop=True)
                    st.success(f"✅ Loaded {len(load_df):,} hours of {country} grid data")

                elif resp.status_code == 401:
                    st.error("API key unauthorised. Please check your ENTSO-E security token.")
                    st.stop()
                elif resp.status_code == 400:
                    st.error(f"Bad request — the API returned: {resp.text[:300]}")
                    st.stop()
                else:
                    st.error(f"ENTSO-E API error {resp.status_code}: {resp.text[:200]}")
                    st.stop()

            except requests.exceptions.Timeout:
                st.error("Request timed out. ENTSO-E API may be slow — try a shorter date range.")
                st.stop()
            except Exception as e:
                st.error(f"Error fetching grid data: {e}")
                st.stop()

        # ── Step 2: Fetch weather from Open-Meteo ────────────────────────
        with st.spinner(f"Fetching weather data for {country}..."):
            try:
                weather_url = (
                    f"https://archive-api.open-meteo.com/v1/archive"
                    f"?latitude={lat}&longitude={lon}"
                    f"&start_date={start_dt.strftime('%Y-%m-%d')}"
                    f"&end_date={end_dt.strftime('%Y-%m-%d')}"
                    f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
                    f"&timezone=UTC"
                )

                wresp = requests.get(weather_url, timeout=30)

                if wresp.status_code == 200:
                    wdata = wresp.json()
                    weather_df = pd.DataFrame({
                        'time':       pd.to_datetime(wdata['hourly']['time']),
                        'temp_c':     wdata['hourly']['temperature_2m'],
                        'humidity':   wdata['hourly']['relative_humidity_2m'],
                        'wind_speed': wdata['hourly']['wind_speed_10m'],
                    })
                    st.success(f"✅ Loaded weather data ({len(weather_df):,} hours)")
                else:
                    st.error(f"Weather API error: {wresp.status_code}")
                    st.stop()

            except Exception as e:
                st.error(f"Error fetching weather: {e}")
                st.stop()

        # ── Step 3: Merge and predict ─────────────────────────────────────
        with st.spinner("Merging data and generating predictions..."):

            # Strip timezone info and floor to hour so both columns match
            load_df['time']    = pd.to_datetime(load_df['time']).dt.tz_localize(None).dt.floor('h')
            weather_df['time'] = pd.to_datetime(weather_df['time']).dt.tz_localize(None).dt.floor('h')
            merged = pd.merge(load_df, weather_df, on='time', how='inner')
            merged = merged.sort_values('time').reset_index(drop=True)

            if len(merged) == 0:
                st.error("No overlapping data between grid and weather. Try a different date range.")
                st.stop()

            # Add features
            merged['hour']      = merged['time'].dt.hour
            merged['dayofweek'] = merged['time'].dt.dayofweek
            merged['month']     = merged['time'].dt.month
            merged['is_weekend']= (merged['dayofweek'] >= 5).astype(int)
            merged['hour_sin']  = np.sin(2*np.pi*merged['hour']/24)
            merged['hour_cos']  = np.cos(2*np.pi*merged['hour']/24)
            merged['month_sin'] = np.sin(2*np.pi*merged['month']/12)
            merged['month_cos'] = np.cos(2*np.pi*merged['month']/12)
            merged['dow_sin']   = np.sin(2*np.pi*merged['dayofweek']/7)
            merged['dow_cos']   = np.cos(2*np.pi*merged['dayofweek']/7)

            merged['load_lag_1h']           = merged['total_load_actual'].shift(1)
            merged['load_lag_24h']          = merged['total_load_actual'].shift(24)
            merged['load_lag_168h']         = merged['total_load_actual'].shift(168)
            merged['load_rolling_mean_24h'] = merged['total_load_actual'].shift(1).rolling(24).mean()
            merged['load_rolling_std_24h']  = merged['total_load_actual'].shift(1).rolling(24).std()

            merged = merged.dropna(subset=FEATURES)

            if len(merged) == 0:
                st.warning("Not enough data rows after lag feature construction. Try fetching more days.")
                st.stop()

            merged['predicted'] = rf_model.predict(merged[FEATURES])

        # ── Step 4: Show results ──────────────────────────────────────────
        st.divider()
        st.subheader(f"📊 Results — {country} ({start_dt.date()} to {end_dt.date()})")

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae  = mean_absolute_error(merged['total_load_actual'], merged['predicted'])
        rmse = np.sqrt(mean_squared_error(merged['total_load_actual'], merged['predicted']))
        r2   = r2_score(merged['total_load_actual'], merged['predicted'])
        mape = ((merged['total_load_actual'] - merged['predicted']).abs() /
                 merged['total_load_actual']).mean() * 100

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Hours",  f"{len(merged):,}")
        c2.metric("MAE",    f"{mae:.0f} MW")
        c3.metric("R²",     f"{r2:.4f}")
        c4.metric("MAPE",   f"{mape:.2f}%")

        st.info(f"ℹ️ Note: The model was trained on Spanish data (2015–2018). "
                f"Performance on {country} data reflects how well Spanish demand patterns "
                f"generalise to other European grids — a genuine out-of-sample test.")

        st.divider()

        # Actual vs Predicted chart
        st.subheader("📈 Actual vs Predicted Load")
        fig, ax = plt.subplots(figsize=(14,4))
        ax.plot(merged['time'], merged['total_load_actual'],
                label='Actual', color='black', linewidth=1.2)
        ax.plot(merged['time'], merged['predicted'],
                label=f'Predicted (R²={r2:.4f})', color='steelblue',
                linewidth=1.2, alpha=0.85)
        ax.fill_between(merged['time'],
                        merged['predicted']-mae, merged['predicted']+mae,
                        alpha=0.15, color='steelblue', label=f'±MAE ({mae:.0f} MW)')
        ax.set_xlabel('Date'); ax.set_ylabel('Load (MW)')
        ax.set_title(f'{country} — Actual vs Predicted Load', fontsize=13)
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.xticks(rotation=30); plt.tight_layout()
        st.pyplot(fig); plt.close()

        # Download
        st.divider()
        dl = merged[['time','total_load_actual','predicted']].copy()
        dl.columns = ['Datetime','Actual Load (MW)','Predicted Load (MW)']
        st.download_button("📥 Download Predictions as CSV",
                           dl.to_csv(index=False),
                           file_name=f"{country.lower()}_predictions.csv",
                           mime="text/csv",
                           use_container_width=True)
