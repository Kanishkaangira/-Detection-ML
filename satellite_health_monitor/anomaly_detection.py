from sklearn.ensemble import IsolationForest


def detect_anomalies(data, contamination=0.05):

    features = data.drop(columns=["time"])

    model = IsolationForest(
        contamination=contamination,
        random_state=None
    )

    data["anomaly"] = model.fit_predict(features)
    data["anomaly_score"] = model.decision_function(features)

    return data