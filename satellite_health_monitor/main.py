from data_simulation import generate_telemetry, inject_faults
from feature_engineering import engineer_features
from anomaly_detection import detect_anomalies
from fault_management import classify_fault, compute_health_index
from logger import log_anomalies
from visualization import plot_dashboard, plot_anomaly_score


def main():

    print("Generating telemetry...")
    data = generate_telemetry()

    print("Injecting synthetic faults...")
    data = inject_faults(data)

    print("Engineering features...")
    data = engineer_features(data)

    print("Running anomaly detection...")
    data = detect_anomalies(data)

    print("Classifying faults...")
    data["fault_type"] = data.apply(classify_fault, axis=1)

    print("Computing health index...")
    data = compute_health_index(data)

    print("Logging anomalies...")
    log_anomalies(data)

    print("Visualizing results...")
    plot_dashboard(data)
    plot_anomaly_score(data)

    print("System Monitoring Completed Successfully.")


if __name__ == "__main__":
    main()