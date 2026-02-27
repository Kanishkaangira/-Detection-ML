def engineer_features(data):

    # Derived features
    data["power"] = data["voltage"] * data["current"]
    data["temp_rate"] = data["temperature"].diff()
    data["rolling_temp_mean"] = data["temperature"].rolling(5).mean()
    data["rolling_temp_std"] = data["temperature"].rolling(5).std()

    data = data.fillna(0)

    return data