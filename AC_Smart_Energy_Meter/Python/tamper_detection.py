import pandas as pd

def detect_tamper(row):
    """Rule-based detection over simulated digital indicators and measurements."""
    alerts = []
    if int(row.get("enclosure_open", 0)) == 1:
        alerts.append("Enclosure opening signal")
    if int(row.get("magnetic_event", 0)) == 1:
        alerts.append("Simulated magnetic interference")
    if int(row.get("bypass_event", 0)) == 1:
        alerts.append("Possible current bypass signature")
    if int(row.get("reverse_event", 0)) == 1 or float(row.get("real_power", 0)) < -1:
        alerts.append("Reverse-current signature")
    if alerts:
        return "TAMPER ALERT", "; ".join(alerts)
    return "NORMAL", "No simulated tamper indicator"

def add_tamper_results(df):
    results = df.apply(detect_tamper, axis=1, result_type="expand")
    results.columns = ["tamper_status", "tamper_reason"]
    return pd.concat([df.reset_index(drop=True), results], axis=1)
