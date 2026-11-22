from pathlib import Path
import sys
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT/"Python"))
from tamper_detection import add_tamper_results

st.set_page_config(page_title="AC Smart Energy Meter", page_icon="⚡", layout="wide")
st.title("⚡ AC Smart Energy Meter")
st.caption("Software simulation | NILM | Anomaly Detection | Tamper Detection")

data_path = ROOT/"Dataset"/"ac_readings.csv"
wave_path = ROOT/"Dataset"/"waveform_data.csv"
nilm_path = ROOT/"Python"/"nilm_model.pkl"
anomaly_path = ROOT/"Python"/"anomaly_model.pkl"

required = [data_path, wave_path, nilm_path, anomaly_path]
missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
if missing:
    st.error("Required project files are missing: " + ", ".join(missing))
    st.info("Run Python/generate_ac_data.py and Python/train_models.py first.")
    st.stop()

df = pd.read_csv(data_path)
waves = pd.read_csv(wave_path)
features = ["voltage_rms","current_rms","frequency","real_power",
            "apparent_power","reactive_power","power_factor"]
nilm = joblib.load(nilm_path)
anomaly_model = joblib.load(anomaly_path)

df["predicted_load"] = nilm.predict(df[features])
df["anomaly_prediction"] = anomaly_model.predict(df[features])
df["anomaly_status"] = np.where(df["anomaly_prediction"] == -1, "ANOMALY", "NORMAL")
df = add_tamper_results(df)

st.sidebar.header("Simulation controls")
load_filter = st.sidebar.multiselect("Load type", sorted(df["load_label"].unique()),
                                     default=sorted(df["load_label"].unique()))
tamper_filter = st.sidebar.selectbox("Tamper events", ["All", "Only tamper alerts", "No tamper alerts"])
view = df[df["load_label"].isin(load_filter)].copy()
if tamper_filter == "Only tamper alerts":
    view = view[view["tamper_status"] == "TAMPER ALERT"]
elif tamper_filter == "No tamper alerts":
    view = view[view["tamper_status"] == "NORMAL"]
if view.empty:
    st.warning("No readings match these filters.")
    st.stop()

latest = view.iloc[-1]
st.subheader("Electrical measurements")
cols = st.columns(4)
cols[0].metric("Voltage (RMS)", f"{latest.voltage_rms:.2f} V")
cols[1].metric("Current (RMS)", f"{latest.current_rms:.3f} A")
cols[2].metric("Frequency", f"{latest.frequency:.2f} Hz")
cols[3].metric("Power factor", f"{latest.power_factor:.3f}")
cols = st.columns(4)
cols[0].metric("Real power", f"{latest.real_power:.2f} W")
cols[1].metric("Apparent power", f"{latest.apparent_power:.2f} VA")
cols[2].metric("Reactive power", f"{latest.reactive_power:.2f} VAR")
energy_kwh = (view["real_power"].clip(lower=0).sum() / 3600) / 1000
tariff = st.sidebar.number_input("Tariff (₹/kWh)", min_value=0.0, value=8.0, step=0.5)
cols[3].metric("Simulated energy", f"{energy_kwh:.6f} kWh")
st.metric("Estimated cost", f"₹{energy_kwh*tariff:.4f}")

st.subheader("Voltage and current waveforms")
reading_id = st.selectbox("Select reading ID", view["reading_id"].astype(int).tolist(),
                          index=len(view)-1)
selected = waves[waves["reading_id"] == reading_id]
if not selected.empty:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(selected["time_seconds"]*1000, selected["voltage_sample"], label="Voltage (V)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (V)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    st.pyplot(fig)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(selected["time_seconds"]*1000, selected["current_sample"], label="Current (A)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Current (A)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Load identification (NILM)")
st.dataframe(view[["reading_id","load_label","predicted_load","voltage_rms",
                   "current_rms","real_power","power_factor"]].tail(20),
             use_container_width=True)

st.subheader("Anomaly detection")
a, b = st.columns(2)
a.metric("Anomalies flagged by model", int((view["anomaly_status"]=="ANOMALY").sum()))
b.metric("Normal readings", int((view["anomaly_status"]=="NORMAL").sum()))
st.bar_chart(view["anomaly_status"].value_counts())

st.subheader("Tamper detection")
t1, t2 = st.columns(2)
t1.metric("Simulated tamper alerts", int((view["tamper_status"]=="TAMPER ALERT").sum()))
t2.metric("Normal tamper status", int((view["tamper_status"]=="NORMAL").sum()))
alerts = view[view["tamper_status"]=="TAMPER ALERT"]
if not alerts.empty:
    st.warning("Simulated tamper events detected. These are generated test cases, not physical tamper measurements.")
    st.dataframe(alerts[["reading_id","tamper_type","tamper_reason","tamper_status"]].tail(20),
                 use_container_width=True)
else:
    st.success("No tamper events in the selected readings.")

st.subheader("Frequency and power trends")
trend = view.set_index("reading_id")[["frequency","real_power","voltage_rms","current_rms"]]
st.line_chart(trend)
st.caption("All displayed data is synthetic. The sample records represent simulated one-second intervals; waveforms show two AC cycles per selected record.")
