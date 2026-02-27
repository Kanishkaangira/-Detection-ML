import matplotlib.pyplot as plt


def plot_dashboard(data):

    fig, axs = plt.subplots(4, 1, figsize=(12, 12))

    axs[0].plot(data["time"], data["temperature"])
    axs[0].set_title("Temperature")

    axs[1].plot(data["time"], data["voltage"])
    axs[1].set_title("Voltage")

    axs[2].plot(data["time"], data["current"])
    axs[2].set_title("Current")

    axs[3].plot(data["time"], data["health_index"])
    axs[3].set_title("Health Index (0-100)")

    plt.tight_layout()
    plt.show()


def plot_anomaly_score(data):

    plt.figure(figsize=(12,5))
    plt.plot(data["time"], data["anomaly_score"])
    plt.axhline(y=0, linestyle="--")
    plt.title("Anomaly Score Over Time")
    plt.xlabel("Time")
    plt.ylabel("Anomaly Score")
    plt.show()