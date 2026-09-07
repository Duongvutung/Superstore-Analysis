from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("Data/Superstore Sale Dataset.csv")
OUTPUT_DIR = Path("Outputs")

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1")   
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%m/%d/%Y")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%m/%d/%Y")
    return df

def find_optimal_k(df: pd.DataFrame, max_k: int = 10):
    customer_df = df.groupby(["Customer ID"], as_index=False).agg(
        OrderCount=("Order ID", "nunique"),
        TotalSales=("Sales", "sum"),
        TotalProfit=("Profit", "sum"),
        TotalQuantity=("Quantity", "sum"),
        AvgDiscount=("Discount", "mean"),
        AvgSalesPerOrder=("Sales", "mean"),
    )
    feature_cols = ["OrderCount", "TotalSales", "TotalProfit", "TotalQuantity", "AvgDiscount", "AvgSalesPerOrder"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(customer_df[feature_cols])

    distortions = []
    for k in range(1, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X_scaled)
        distortions.append(model.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), distortions, 'bx-')
    plt.xlabel('Số lượng cụm (k)')
    plt.ylabel('Sai số (Inertia)')
    plt.title('Phương pháp Elbow trong việc chọn cụm k tối ưu')
    plt.grid(True)
    plt.show()

def anomaly_detection(df: pd.DataFrame, contamination: float = 0.03):
    feature_cols = ["Sales", "Profit", "Discount", "Quantity"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])

    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    preds = model.fit_predict(X_scaled)
    scores = model.decision_function(X_scaled)

    result = df.copy()
    result["AnomalyLabel"] = ["Anomaly" if p == -1 else "Normal" for p in preds]
    result["AnomalyScore"] = scores

    anomaly_only = result[result["AnomalyLabel"] == "Anomaly"].sort_values("AnomalyScore")
    summary = pd.DataFrame({
        "Contamination": [contamination],
        "TotalRows": [len(result)],
        "AnomalyRows": [len(anomaly_only)],
        "AnomalyRatePercent": [len(anomaly_only) / len(result) * 100 if len(result) else 0],
    })
    return result, anomaly_only, summary

def customer_segmentation(df: pd.DataFrame, n_clusters: int = 4):
    customer_df = df.groupby(["Customer ID", "Customer Name"], as_index=False).agg(
        OrderCount=("Order ID", "nunique"),
        TotalSales=("Sales", "sum"),
        TotalProfit=("Profit", "sum"),
        TotalQuantity=("Quantity", "sum"),
        AvgDiscount=("Discount", "mean"),
        AvgSalesPerOrder=("Sales", "mean"),
    )

    feature_cols = ["OrderCount", "TotalSales", "TotalProfit", "TotalQuantity", "AvgDiscount", "AvgSalesPerOrder"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(customer_df[feature_cols])

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    customer_df["ClusterId"] = model.fit_predict(X_scaled)

    # Đặt tên cụm dựa trên trung bình doanh thu
    cluster_rank = customer_df.groupby("ClusterId", as_index=False)["TotalSales"].mean().sort_values("TotalSales", ascending=False).reset_index(drop=True)
    labels = ["High Value", "Medium Value", "Low Value", "Emerging Value"]
    name_map = {row["ClusterId"]: labels[idx] if idx < len(labels) else f"Cluster {idx + 1}" for idx, row in cluster_rank.iterrows()}
    customer_df["ClusterName"] = customer_df["ClusterId"].map(name_map)

    summary = customer_df.groupby("ClusterName", as_index=False).agg(
        Customers=("Customer ID", "count"),
        TotalSales=("TotalSales", "sum"),
        TotalProfit=("TotalProfit", "sum"),
        TotalQuantity=("TotalQuantity", "sum"),
        AvgDiscount=("AvgDiscount", "mean"),
    ).sort_values("TotalSales", ascending=False)
    
    return customer_df, summary

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing CSV file: {DATA_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(DATA_PATH)

    # 1. Chạy phân tích Elbow
    find_optimal_k(df)

    # 2. Phân tích chính
    all_rows, anomaly_rows, anomaly_summary = anomaly_detection(df, contamination=0.03)
    customer_df, cluster_summary = customer_segmentation(df, n_clusters=4)

    # Lưu file
    all_rows.to_csv(OUTPUT_DIR / "anomaly_labeled_transactions.csv", index=False)
    customer_df.to_csv(OUTPUT_DIR / "customer_segments.csv", index=False)
    cluster_summary.to_csv(OUTPUT_DIR / "customer_segment_summary.csv", index=False)
    
    print("Đã hoàn tất xử lý. Kết quả đã lưu tại:", OUTPUT_DIR)

if __name__ == "__main__":
    main()