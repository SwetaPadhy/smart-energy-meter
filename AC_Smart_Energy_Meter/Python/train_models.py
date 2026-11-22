from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"Dataset"/"ac_readings.csv"
OUT = ROOT/"Python"
df = pd.read_csv(DATA)

features = ["voltage_rms","current_rms","frequency","real_power",
            "apparent_power","reactive_power","power_factor"]
X, y = df[features], df["load_label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
nilm = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
nilm.fit(X_train, y_train)
pred = nilm.predict(X_test)
joblib.dump(nilm, OUT/"nilm_model.pkl")
print("NILM test accuracy on synthetic holdout data:", round(accuracy_score(y_test,pred),3))
print(classification_report(y_test,pred))

# Fit anomaly detector mostly on readings labelled normal.
normal = df[df["anomaly_label"]=="NORMAL"][features]
anomaly = IsolationForest(n_estimators=200, contamination=0.06, random_state=42)
anomaly.fit(normal)
joblib.dump(anomaly, OUT/"anomaly_model.pkl")
print("Saved NILM and anomaly models.")
