from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

DATA_PATH = Path("Data/Superstore Sale Dataset.csv")
OUTPUT_DIR = Path("Outputs")

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1")   
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%m/%d/%Y")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%m/%d/%Y")
    return df

def find_optimal_k(df: pd.DataFrame, max_k: int = 8):
    customer_df = df.groupby(["Customer ID"], as_index=False).agg(
        OrderCount=("Order ID", "nunique"),
        TotalSales=("Sales", "sum"),
        TotalProfit=("Profit", "sum"),
        AvgDiscount=("Discount", "mean"),
    )
    feature_cols = ["OrderCount", "TotalSales", "TotalProfit", "AvgDiscount"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(customer_df[feature_cols])

    distortions = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        distortions.append(model.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))

    print("=== ĐÁNH GIÁ CHỌN CỤM K TỐI ƯU ===")
    for k, s in zip(k_range, silhouette_scores):
        print(f"K = {k} | Silhouette Score: {s:.4f}")

    return customer_df, feature_cols

def anomaly_detection(df: pd.DataFrame, contamination: float = 0.03):
    """
    Phát hiện các giao dịch dị biệt (extreme transaction outliers)
    dựa trên Doanh thu, Lợi nhuận, Chiết khấu và Số lượng.
    """
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
        "TotalTransactions": [len(result)],
        "AnomalyTransactions": [len(anomaly_only)],
        "AnomalyRatePercent": [round(len(anomaly_only) / len(result) * 100, 2)],
        "NegativeProfitAnomalies": [len(anomaly_only[anomaly_only["Profit"] < 0])],
        "HighDiscountAnomalies": [len(anomaly_only[anomaly_only["Discount"] >= 0.5])],
    })
    return result, anomaly_only, summary

def customer_segmentation(df: pd.DataFrame, n_clusters: int = 4):
    """
    Phân cụm khách hàng theo RFM / Business Metrics và gán hành động nghiệp vụ
    """
    customer_df = df.groupby(["Customer ID", "Customer Name"], as_index=False).agg(
        OrderCount=("Order ID", "nunique"),
        TotalSales=("Sales", "sum"),
        TotalProfit=("Profit", "sum"),
        AvgDiscount=("Discount", "mean"),
    )

    feature_cols = ["OrderCount", "TotalSales", "TotalProfit", "AvgDiscount"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(customer_df[feature_cols])

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    customer_df["ClusterId"] = model.fit_predict(X_scaled)

    # Đặt tên và gán hành động cụ thể cho từng cụm
    cluster_rank = customer_df.groupby("ClusterId", as_index=False)["TotalSales"].mean().sort_values("TotalSales", ascending=False).reset_index(drop=True)
    
    # Định nghĩa nhãn & hành động kinh doanh tương ứng
    business_actions = [
        {"Name": "VIP / Champions", "Action": "Chăm sóc đặc biệt, ưu đãi độc quyền để giữ chân"},
        {"Name": "Potential Loyal", "Action": "Cross-sell, tăng hạn mức mua sắm"},
        {"Name": "At Risk / Price Sensitive", "Action": "Tối ưu hóa chiết khấu, rà soát biên lợi nhuận"},
        {"Name": "Standard / Low Engagement", "Action": "Duy trì email marketing định kỳ, tránh đốt chi phí"}
    ]

    cluster_map = {}
    action_map = {}
    for idx, row in cluster_rank.iterrows():
        c_id = row["ClusterId"]
        if idx < len(business_actions):
            cluster_map[c_id] = business_actions[idx]["Name"]
            action_map[c_id] = business_actions[idx]["Action"]
        else:
            cluster_map[c_id] = f"Cluster {idx + 1}"
            action_map[c_id] = "Theo dõi thêm"

    customer_df["ClusterName"] = customer_df["ClusterId"].map(cluster_map)
    customer_df["ActionStrategy"] = customer_df["ClusterId"].map(action_map)

    summary = customer_df.groupby(["ClusterName", "ActionStrategy"], as_index=False).agg(
        Customers=("Customer ID", "count"),
        TotalSales=("TotalSales", "sum"),
        TotalProfit=("TotalProfit", "sum"),
        AvgDiscount=("AvgDiscount", "mean"),
    ).sort_values("TotalSales", ascending=False)
    
    return customer_df, summary

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing CSV file: {DATA_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(DATA_PATH)

    # 1. Chạy đánh giá Silhouette / Elbow
    find_optimal_k(df)

    # 2. Phân tích dị biệt & lưu đủ bảng tổng hợp (sửa lỗi thiếu file xuất trước đây)
    all_rows, anomaly_rows, anomaly_summary = anomaly_detection(df, contamination=0.03)
    customer_df, cluster_summary = customer_segmentation(df, n_clusters=4)

    # Xuất dữ liệu
    all_rows.to_csv(OUTPUT_DIR / "anomaly_labeled_transactions.csv", index=False)
    anomaly_rows.to_csv(OUTPUT_DIR / "anomalies_only.csv", index=False)
    anomaly_summary.to_csv(OUTPUT_DIR / "anomaly_summary.csv", index=False)
    customer_df.to_csv(OUTPUT_DIR / "customer_segments.csv", index=False)
    cluster_summary.to_csv(OUTPUT_DIR / "customer_segment_summary.csv", index=False)
    
    print("Đã hoàn tất xử lý và xuất đầy đủ báo cáo tại thư mục:", OUTPUT_DIR)

if __name__ == "__main__":
    main()