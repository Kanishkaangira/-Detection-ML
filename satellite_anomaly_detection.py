import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(None)

# Simulated time steps
time = np.arange(0, 200)

# Normal satellite temperature (around 30°C)
temperature = np.random.normal(30, 2, 200)

# Normal voltage (around 5V)
voltage = np.random.normal(5, 0.2, 200)

# Create dataframe
data = pd.DataFrame({
    "time": time,
    "temperature": temperature,
    "voltage": voltage
})

# Randomly inject 5 anomalies
anomaly_indices = np.random.choice(range(200), 5, replace=False)

for idx in anomaly_indices:
    if np.random.rand() > 0.5:
        data.loc[idx, "temperature"] = np.random.choice([5, 85])
    else:
        data.loc[idx, "voltage"] = np.random.choice([1, 8])

from sklearn.ensemble import IsolationForest

# Select features
features = data[["temperature", "voltage"]]

# Train anomaly detection model
model = IsolationForest(contamination=0.05,random_state=None)
data["anomaly"] = model.fit_predict(features)

data["anomaly_score"] = model.decision_function(features)

plt.figure(figsize=(12,6))

# Normal points
normal = data[data["anomaly"] == 1]
plt.scatter(normal["time"], normal["temperature"],
            color="blue", label="Normal")

# Anomaly points
anomaly = data[data["anomaly"] == -1]
plt.scatter(anomaly["time"], anomaly["temperature"],
            color="red", label="Anomaly", s=100)

plt.title("Satellite Telemetry Health Monitoring")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.legend()
plt.show()

plt.figure(figsize=(12,5))
plt.plot(data["time"], data["anomaly_score"])
plt.title("Anomaly Score Over Time")
plt.xlabel("Time")
plt.ylabel("Anomaly Score")
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

from scipy import stats

z_scores = np.abs(stats.zscore(data[["temperature", "voltage"]]))
data["stat_anomaly"] = (z_scores > 3).any(axis=1)

print("Number of anomalies detected:", 
      len(data[data["anomaly"] == -1]))

print("Statistical detected anomalies:",
      data["stat_anomaly"].sum())

print("Correlation Matrix:")
print(data[["temperature", "voltage"]].corr())

plt.figure(figsize=(12,6))

normal = data[data["anomaly"] == 1]
plt.scatter(normal["time"], normal["voltage"],
            color="blue", label="Normal")

anomaly = data[data["anomaly"] == -1]
plt.scatter(anomaly["time"], anomaly["voltage"],
            color="red", label="Anomaly", s=100)

plt.title("Voltage Anomaly Detection")
plt.xlabel("Time")
plt.ylabel("Voltage")
plt.legend()
plt.show()
