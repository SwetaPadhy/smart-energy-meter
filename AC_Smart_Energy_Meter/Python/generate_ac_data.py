from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Dataset"
DATA.mkdir(exist_ok=True)

def generate():
    rng = np.random.default_rng(42)
    fs = 10000
    t = np.arange(400) / fs             # 2 cycles at nominal 50 Hz
    classes = ["OFF", "LED", "FAN", "MULTIPLE"]
    settings = {
        "OFF": (0.015, 0.50),
        "LED": (0.10, 0.96),
        "FAN": (0.35, 0.78),
        "MULTIPLE": (0.62, 0.88),
    }
    rows, wave_rows = [], []
    rid = 0
    for label in classes:
        for _ in range(125):
            rid += 1
            vrms = rng.normal(230, 2.0)
            freq = rng.normal(50, 0.12)
            irms, pf = settings[label]
            irms = max(0.005, rng.normal(irms, max(0.004, irms*0.06)))
            pf = float(np.clip(rng.normal(pf, 0.025), 0.2, 0.999))
            phase = np.arccos(pf)
            # Slight 3rd harmonic to make the simulated signal less ideal.
            phi_v = rng.uniform(-0.02, 0.02)
            v = np.sqrt(2)*vrms*np.sin(2*np.pi*freq*t + phi_v)
            i = np.sqrt(2)*irms*np.sin(2*np.pi*freq*t + phi_v - phase)
            i += 0.025*np.sqrt(2)*irms*np.sin(2*np.pi*3*freq*t + phi_v - phase)
            tamper_type = "NONE"
            enclosure_open = 0
            magnetic_event = 0
            bypass_event = 0
            reverse_event = 0

            # About 12% simulated tamper cases, distributed across event types.
            if rng.random() < 0.12:
                tamper_type = rng.choice(["ENCLOSURE_OPEN", "MAGNETIC_INTERFERENCE",
                                          "CURRENT_BYPASS", "REVERSE_CURRENT"])
                if tamper_type == "ENCLOSURE_OPEN":
                    enclosure_open = 1
                elif tamper_type == "MAGNETIC_INTERFERENCE":
                    magnetic_event = 1
                    i *= rng.uniform(0.35, 0.65)
                elif tamper_type == "CURRENT_BYPASS":
                    bypass_event = 1
                    i *= 0.12
                elif tamper_type == "REVERSE_CURRENT":
                    reverse_event = 1
                    i *= -1

            # Sensor-like noise; still entirely synthetic.
            v += rng.normal(0, 0.15, size=len(t))
            i += rng.normal(0, max(0.0005, irms*0.012), size=len(t))
            v_rms = float(np.sqrt(np.mean(v**2)))
            i_rms = float(np.sqrt(np.mean(i**2)))
            p = float(np.mean(v*i))
            s = float(v_rms*i_rms)
            q = float(np.sqrt(max(0.0, s*s - p*p)))
            pf_calc = float(np.clip(p/s if s else 0.0, -1, 1))
            # Each row represents one simulated second for energy accumulation.
            rows.append({
                "reading_id": rid, "voltage_rms": v_rms, "current_rms": i_rms,
                "frequency": freq, "real_power": p, "apparent_power": s,
                "reactive_power": q, "power_factor": pf_calc,
                "load_label": label, "tamper_type": tamper_type,
                "enclosure_open": enclosure_open, "magnetic_event": magnetic_event,
                "bypass_event": bypass_event, "reverse_event": reverse_event,
                "duration_seconds": 1.0
            })
            for k, (tv, ci) in enumerate(zip(v, i)):
                wave_rows.append({"reading_id": rid, "sample": k,
                                  "time_seconds": float(t[k]),
                                  "voltage_sample": float(tv),
                                  "current_sample": float(ci)})
    df = pd.DataFrame(rows)
    # A small number of unusual electrical conditions for anomaly-model training.
    df["anomaly_label"] = "NORMAL"
    abnormal_idx = rng.choice(df.index, size=24, replace=False)
    for idx in abnormal_idx:
        df.loc[idx, "anomaly_label"] = "ANOMALY"
        df.loc[idx, "voltage_rms"] *= rng.choice([0.72, 1.25])
        df.loc[idx, "frequency"] *= rng.choice([0.88, 1.12])
        df.loc[idx, "current_rms"] *= rng.choice([0.35, 2.0])
        df.loc[idx, "real_power"] = df.loc[idx, "voltage_rms"] * df.loc[idx, "current_rms"] * df.loc[idx, "power_factor"]
        df.loc[idx, "apparent_power"] = df.loc[idx, "voltage_rms"] * df.loc[idx, "current_rms"]
    df.to_csv(DATA/"ac_readings.csv", index=False)
    pd.DataFrame(wave_rows).to_csv(DATA/"waveform_data.csv", index=False)
    print(f"Generated {len(df)} summary readings and {len(wave_rows)} waveform samples.")
    print(f"Tamper events: {(df.tamper_type != 'NONE').sum()} | Anomaly labels: {(df.anomaly_label == 'ANOMALY').sum()}")

if __name__ == "__main__":
    generate()
