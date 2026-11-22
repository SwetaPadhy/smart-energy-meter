import numpy as np

def calculate_ac_parameters(voltage_samples, current_samples, sample_rate):
    v = np.asarray(voltage_samples, dtype=float)
    i = np.asarray(current_samples, dtype=float)
    if len(v) != len(i) or len(v) < 4:
        raise ValueError("Voltage and current arrays must have equal length and at least 4 samples.")
    vrms = float(np.sqrt(np.mean(v**2)))
    irms = float(np.sqrt(np.mean(i**2)))
    real_power = float(np.mean(v*i))
    apparent_power = float(vrms*irms)
    reactive_power = float(np.sqrt(max(0.0, apparent_power**2-real_power**2)))
    pf = float(real_power/apparent_power) if apparent_power else 0.0
    centered = v - np.mean(v)
    crossings = np.where((centered[:-1] <= 0) & (centered[1:] > 0))[0]
    if len(crossings) >= 2:
        periods = np.diff(crossings)/float(sample_rate)
        frequency = float(1/np.mean(periods))
    else:
        frequency = float("nan")
    return {"voltage_rms":vrms, "current_rms":irms, "frequency":frequency,
            "real_power":real_power, "apparent_power":apparent_power,
            "reactive_power":reactive_power, "power_factor":pf}
