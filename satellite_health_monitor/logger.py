def log_anomalies(data):

    anomalies = data[data["anomaly"] == -1]

    anomalies.to_csv("anomaly_log.csv", index=False)

    print(f"Total anomalies detected: {len(anomalies)}")
    print("Anomaly log saved to anomaly_log.csv")