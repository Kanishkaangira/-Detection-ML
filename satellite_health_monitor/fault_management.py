def classify_fault(row):

    if row["temperature"] > 60:
        return "Thermal Fault"

    elif row["voltage"] > 6:
        return "Power Fault"

    elif abs(row["gyro_x"]) > 0.5:
        return "Attitude Fault"

    elif row["signal_strength"] < 50:
        return "Communication Fault"

    else:
        return "Normal"


def compute_health_index(data):

    # Convert anomaly score to 0–100 scale
    data["health_index"] = 100 + (data["anomaly_score"] * 50)
    data["health_index"] = data["health_index"].clip(0, 100)

    return data