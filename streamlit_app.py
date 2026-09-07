import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Superstore Sales Dashboard + AI",
    page_icon="📊",
    layout="wide",
)


DATA_PATH = Path("Data/Superstore Sale Dataset.csv")
MODEL_PATH = Path("Outputs/linear_regression_sales_model.pkl")


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1")
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%m/%d/%Y")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%m/%d/%Y")
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month_name()
    df["Month Number"] = df["Order Date"].dt.month
    df["YearMonth"] = df["Order Date"].dt.to_period("M").astype(str)
    df["MonthStart"] = df["Order Date"].values.astype("datetime64[M]")
    return df


@st.cache_resource
def load_model(path: Path):
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


@st.cache_data
def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    customer_df = (
        df.groupby(["Customer ID", "Customer Name"], as_index=False)
        .agg(
            OrderCount=("Order ID", "nunique"),
            TotalSales=("Sales", "sum"),
            TotalProfit=("Profit", "sum"),
            TotalQuantity=("Quantity", "sum"),
            AvgDiscount=("Discount", "mean"),
            AvgSalesPerOrder=("Sales", "mean"),
        )
        .sort_values("TotalSales", ascending=False)
        .reset_index(drop=True)
    )
    return customer_df


@st.cache_data
def detect_anomalies(df: pd.DataFrame, contamination: float) -> pd.DataFrame:
    work = df.copy()
    feature_cols = ["Sales", "Profit", "Discount", "Quantity"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(work[feature_cols])

    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=200,
    )
    preds = model.fit_predict(X_scaled)
    scores = model.decision_function(X_scaled)

    work["AnomalyLabel"] = np.where(preds == -1, "Anomaly", "Normal")
    work["AnomalyScore"] = scores
    return work.sort_values(["AnomalyLabel", "AnomalyScore"], ascending=[False, True])


@st.cache_data
def segment_customers(df: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    customer_df = build_customer_features(df)
    feature_cols = [
        "OrderCount",
        "TotalSales",
        "TotalProfit",
        "TotalQuantity",
        "AvgDiscount",
        "AvgSalesPerOrder",
    ]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(customer_df[feature_cols])

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    customer_df["ClusterId"] = model.fit_predict(X_scaled)

    cluster_rank = (
        customer_df.groupby("ClusterId", as_index=False)["TotalSales"]
        .mean()
        .sort_values("TotalSales", ascending=False)
        .reset_index(drop=True)
    )

    name_map = {}
    labels = ["High Value", "Medium Value", "Low Value", "Emerging Value", "Potential Value"]
    for idx, row in cluster_rank.iterrows():
        name_map[row["ClusterId"]] = labels[idx] if idx < len(labels) else f"Cluster {idx + 1}"

    customer_df["ClusterName"] = customer_df["ClusterId"].map(name_map)
    return customer_df.sort_values(["ClusterName", "TotalSales"], ascending=[True, False])


@st.cache_data
def generate_ai_insights(df: pd.DataFrame, anomaly_df: pd.DataFrame, customer_df: pd.DataFrame) -> list[str]:
    insights: list[str] = []

    corr = df[["Discount", "Profit"]].corr().loc["Discount", "Profit"]
    insights.append(
        f"Tương quan giữa Discount và Profit là {corr:.3f}. Điều này cho thấy khi chiết khấu tăng, lợi nhuận có xu hướng giảm."
    )

    anomaly_count = int((anomaly_df["AnomalyLabel"] == "Anomaly").sum())
    anomaly_ratio = (anomaly_count / len(anomaly_df) * 100) if len(anomaly_df) else 0
    insights.append(
        f"Hệ thống phát hiện {anomaly_count:,} giao dịch bất thường, tương đương {anomaly_ratio:.2f}% dữ liệu sau lọc."
    )

    cluster_summary = (
        customer_df.groupby("ClusterName", as_index=False)
        .agg(
            Customers=("Customer ID", "count"),
            TotalSales=("TotalSales", "sum"),
            TotalProfit=("TotalProfit", "sum"),
        )
        .sort_values("TotalSales", ascending=False)
    )
    if not cluster_summary.empty:
        top_cluster = cluster_summary.iloc[0]
        insights.append(
            f"Nhóm khách hàng nổi bật nhất là {top_cluster['ClusterName']} với {int(top_cluster['Customers'])} khách hàng, doanh thu {top_cluster['TotalSales']:,.2f} và lợi nhuận {top_cluster['TotalProfit']:,.2f}."
        )

    loss_subcats = (
        df.groupby("Sub-Category", as_index=False)["Profit"]
        .sum()
        .query("Profit < 0")
        .sort_values("Profit")
    )
    if not loss_subcats.empty:
        top_loss = loss_subcats.iloc[0]
        insights.append(
            f"Nhóm sản phẩm cần chú ý nhất là {top_loss['Sub-Category']} vì có tổng lợi nhuận âm {top_loss['Profit']:,.2f}."
        )

    return insights



def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Bộ lọc dữ liệu")

    years = sorted(df["Year"].dropna().unique().tolist())
    regions = sorted(df["Region"].dropna().unique().tolist())
    segments = sorted(df["Segment"].dropna().unique().tolist())
    categories = sorted(df["Category"].dropna().unique().tolist())

    selected_years = st.sidebar.multiselect("Year", years, default=years)
    selected_regions = st.sidebar.multiselect("Region", regions, default=regions)
    selected_segments = st.sidebar.multiselect("Segment", segments, default=segments)
    selected_categories = st.sidebar.multiselect("Category", categories, default=categories)

    filtered = df[
        df["Year"].isin(selected_years)
        & df["Region"].isin(selected_regions)
        & df["Segment"].isin(selected_segments)
        & df["Category"].isin(selected_categories)
    ].copy()

    return filtered



def format_number(value: float) -> str:
    return f"{value:,.2f}"



def render_kpis(df: pd.DataFrame) -> None:
    total_sales = df["Sales"].sum()
    total_profit = df["Profit"].sum()
    total_orders = df["Order ID"].nunique()
    total_quantity = df["Quantity"].sum()
    profit_margin = (total_profit / total_sales * 100) if total_sales else 0
    avg_order_value = (total_sales / total_orders) if total_orders else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Sales", format_number(total_sales))
    c2.metric("Total Profit", format_number(total_profit))
    c3.metric("Total Orders", f"{total_orders:,}")
    c4.metric("Total Quantity", f"{int(total_quantity):,}")
    c5.metric("Profit Margin", f"{profit_margin:.2f}%")
    c6.metric("Average Order Value", format_number(avg_order_value))



def render_overview(df: pd.DataFrame) -> None:
    st.subheader("Trang Tổng quan")
    render_kpis(df)

    monthly_sales = df.groupby("YearMonth", as_index=False)["Sales"].sum()
    monthly_profit = df.groupby("YearMonth", as_index=False)["Profit"].sum()
    sales_by_region = df.groupby("Region", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
    sales_by_category = df.groupby("Category", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
    profit_by_category = df.groupby("Category", as_index=False)["Profit"].sum().sort_values("Profit", ascending=False)
    top_products = df.groupby("Product Name", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False).head(10)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(monthly_sales, x="YearMonth", y="Sales", markers=True, title="Monthly Sales")
        fig.update_layout(xaxis_title="Year-Month", yaxis_title="Sales")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.line(monthly_profit, x="YearMonth", y="Profit", markers=True, title="Monthly Profit")
        fig.update_layout(xaxis_title="Year-Month", yaxis_title="Profit")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.bar(sales_by_region, x="Region", y="Sales", title="Sales by Region")
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        fig = px.bar(sales_by_category, x="Category", y="Sales", title="Sales by Category")
        st.plotly_chart(fig, use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        fig = px.bar(profit_by_category, x="Category", y="Profit", title="Profit by Category")
        st.plotly_chart(fig, use_container_width=True)
    with col6:
        fig = px.bar(top_products, x="Product Name", y="Sales", title="Top Products by Sales")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)



def render_detail(df: pd.DataFrame) -> None:
    st.subheader("Trang Phân tích chi tiết")

    subcat_sales = df.groupby("Sub-Category", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
    subcat_profit = df.groupby("Sub-Category", as_index=False)["Profit"].sum().sort_values("Profit", ascending=False)
    segment_profit = df.groupby("Segment", as_index=False)["Profit"].sum().sort_values("Profit", ascending=False)
    bottom_products = df.groupby("Product Name", as_index=False)["Profit"].sum().sort_values("Profit").head(10)
    loss_products = df.groupby("Product Name", as_index=False)["Profit"].sum().query("Profit < 0").sort_values("Profit")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(
            df,
            x="Discount",
            y="Profit",
            color="Category",
            title="Discount vs Profit",
            hover_data=["Product Name", "Sub-Category", "Sales"],
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(subcat_sales, x="Sub-Category", y="Sales", title="Sales by Sub-Category")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.bar(subcat_profit, x="Sub-Category", y="Profit", title="Profit by Sub-Category")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        fig = px.bar(segment_profit, x="Segment", y="Profit", title="Profit by Segment")
        st.plotly_chart(fig, use_container_width=True)

    fig_bottom = px.bar(bottom_products, x="Product Name", y="Profit", title="Bottom Products by Profit")
    fig_bottom.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_bottom, use_container_width=True)

    with st.expander("Danh sách sản phẩm bị lỗ"):
        st.dataframe(loss_products, use_container_width=True)



def render_forecast(df: pd.DataFrame, model) -> None:
    st.subheader("Trang Dự báo")

    monthly_sales = df.groupby(pd.Grouper(key="Order Date", freq="MS"))["Sales"].sum().reset_index()
    monthly_sales.columns = ["MonthStart", "Sales"]
    monthly_sales["TimeIndex"] = np.arange(len(monthly_sales))

    fig_hist = px.line(monthly_sales, x="MonthStart", y="Sales", markers=True, title="Historical Monthly Sales")
    st.plotly_chart(fig_hist, use_container_width=True)

    if model is None:
        st.info("Chưa tìm thấy file model .pkl. Bạn vẫn có thể hiển thị forecast lịch sử sau khi huấn luyện và lưu model.")
        return

    future_steps = st.slider("Số tháng dự báo", min_value=3, max_value=12, value=6, step=1)
    last_index = int(monthly_sales["TimeIndex"].max())
    future_index = np.arange(last_index + 1, last_index + 1 + future_steps).reshape(-1, 1)
    future_pred = model.predict(future_index)

    last_date = monthly_sales["MonthStart"].max()
    future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=future_steps, freq="MS")
    future_df = pd.DataFrame({"MonthStart": future_dates, "PredictedSales": future_pred})

    fig = px.line(monthly_sales, x="MonthStart", y="Sales", markers=True, title="Historical Sales and Forecast")
    fig.add_scatter(x=future_df["MonthStart"], y=future_df["PredictedSales"], mode="lines+markers", name="Forecast")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(future_df, use_container_width=True)



def render_ai(df: pd.DataFrame) -> None:
    st.subheader("Trang Phân tích mở rộng")
    st.caption("Tích hợp chức năng phát hiện bất thường và phân cụm khách hàng.")

    col1, col2 = st.columns(2)
    with col1:
        contamination_pct = st.slider("Tỷ lệ phát hiện bất thường (%)", min_value=1, max_value=10, value=3, step=1)
    with col2:
        n_clusters = st.slider("Số cụm khách hàng", min_value=3, max_value=5, value=4, step=1)

    anomaly_df = detect_anomalies(df, contamination=contamination_pct / 100)
    anomaly_only = anomaly_df[anomaly_df["AnomalyLabel"] == "Anomaly"].copy()

    customer_cluster_df = segment_customers(df, n_clusters=n_clusters)
    cluster_summary = (
        customer_cluster_df.groupby("ClusterName", as_index=False)
        .agg(
            Customers=("Customer ID", "count"),
            TotalSales=("TotalSales", "sum"),
            TotalProfit=("TotalProfit", "sum"),
            AvgDiscount=("AvgDiscount", "mean"),
        )
        .sort_values("TotalSales", ascending=False)
    )

    insights = generate_ai_insights(df, anomaly_df, customer_cluster_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Số giao dịch bất thường", f"{len(anomaly_only):,}")
    c2.metric("Tỷ lệ bất thường", f"{(len(anomaly_only) / len(df) * 100):.2f}%")
    c3.metric("Số cụm khách hàng", f"{n_clusters}")

    st.markdown("### 1. Phát hiện bất thường")
    fig_anomaly = px.scatter(
        anomaly_df,
        x="Sales",
        y="Profit",
        color="AnomalyLabel",
        hover_data=["Order ID", "Product Name", "Category", "Discount", "Quantity"],
        title="Sales vs Profit with Anomaly Detection",
    )
    st.plotly_chart(fig_anomaly, use_container_width=True)

    anomaly_table = anomaly_only[
        ["Order ID", "Product Name", "Category", "Sub-Category", "Sales", "Profit", "Discount", "Quantity", "AnomalyScore"]
    ].sort_values("AnomalyScore").head(20)
    st.dataframe(anomaly_table, use_container_width=True)

    st.markdown("### 2. Phân cụm khách hàng")
    fig_cluster = px.scatter(
        customer_cluster_df,
        x="TotalSales",
        y="TotalProfit",
        color="ClusterName",
        size="OrderCount",
        hover_data=["Customer Name", "TotalQuantity", "AvgDiscount"],
        title="Customer Segmentation by Sales and Profit",
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

    st.dataframe(cluster_summary, use_container_width=True)

    with st.expander("Danh sách khách hàng theo cụm"):
        st.dataframe(
            customer_cluster_df[
                [
                    "Customer ID",
                    "Customer Name",
                    "ClusterName",
                    "OrderCount",
                    "TotalSales",
                    "TotalProfit",
                    "TotalQuantity",
                    "AvgDiscount",
                ]
            ],
            use_container_width=True,
        )

    st.markdown("### 3. Nhận xét tự động từ dữ liệu")
    for idx, insight in enumerate(insights, start=1):
        st.write(f"{idx}. {insight}")



def main() -> None:
    st.title("Dashboard phân tích dữ liệu bán hàng Superstore")

    if not DATA_PATH.exists():
        st.error("Không tìm thấy file dữ liệu. Hãy đặt file CSV tại đường dẫn: Data/Superstore Sale Dataset.csv")
        st.stop()

    df = load_data(DATA_PATH)
    model = load_model(MODEL_PATH)
    filtered_df = filter_data(df)

    if filtered_df.empty:
        st.warning("Không có dữ liệu phù hợp với bộ lọc hiện tại.")
        st.stop()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Tổng quan", "Phân tích chi tiết", "Dự báo", "Mở rộng", "Dữ liệu"])

    with tab1:
        render_overview(filtered_df)
    with tab2:
        render_detail(filtered_df)
    with tab3:
        render_forecast(filtered_df, model)
    with tab4:
        render_ai(filtered_df)
    with tab5:
        st.subheader("Xem dữ liệu")
        st.dataframe(filtered_df, use_container_width=True)
        st.write("Kích thước dữ liệu sau lọc:", filtered_df.shape)


if __name__ == "__main__":
    main()
