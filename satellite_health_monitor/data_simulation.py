import numpy as np
import pandas as pd


def generate_telemetry(n=300):
    np.random.seed(None)

    time = np.arange(n)

    data = pd.DataFrame({
        "time": time,
        "temperature": np.random.normal(30, 2, n),
        "voltage": np.random.normal(5, 0.2, n),
        "current": np.random.normal(2, 0.1, n),
        "gyro_x": np.random.normal(0, 0.05, n),
        "gyro_y": np.random.normal(0, 0.05, n),
        "signal_strength": np.random.normal(80, 5, n)
    })

    return data


def inject_faults(data, num_faults=10):
    indices = np.random.choice(len(data), num_faults, replace=False)

    for idx in indices:
        fault = np.random.choice(["thermal", "power", "attitude", "comm"])

        if fault == "thermal":
            data.loc[idx, "temperature"] = 90

        elif fault == "power":
            data.loc[idx, "voltage"] = 8

        elif fault == "attitude":
            data.loc[idx, "gyro_x"] = 1

        elif fault == "comm":
            data.loc[idx, "signal_strength"] = 30

    return data