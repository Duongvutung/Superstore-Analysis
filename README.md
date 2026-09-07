# Đồ án phân tích dữ liệu bán hàng Superstore

## Giới thiệu
Đây là đồ án tốt nghiệp với đề tài **“Nghiên cứu và phát triển Dashboard phân tích dữ liệu bán hàng dựa trên bộ dữ liệu Superstore”**.  
Dự án kết hợp giữa **Power BI**, **Python**, **Machine Learning cơ bản** và **Streamlit** nhằm hỗ trợ trực quan hóa dữ liệu, phân tích dữ liệu bán hàng, dự báo doanh thu, phát hiện bất thường và phân cụm khách hàng.

## Mục tiêu chính
- Xây dựng dashboard phân tích dữ liệu bán hàng trên bộ dữ liệu Superstore
- Theo dõi các KPI quan trọng như doanh thu, lợi nhuận, số đơn hàng, số lượng bán
- Phân tích dữ liệu theo nhiều chiều bằng Power BI và Python
- Dự báo doanh thu bằng mô hình hồi quy tuyến tính
- Phát hiện bất thường trong dữ liệu bán hàng bằng Isolation Forest
- Phân cụm khách hàng bằng K-Means
- Triển khai ứng dụng web bằng Streamlit để tích hợp kết quả

## Cấu trúc thư mục
- `Data/`: chứa dữ liệu đầu vào
- `Notebooks/`: chứa notebook phân tích dữ liệu, EDA và dự báo
- `Outputs/`: chứa các file kết quả đầu ra như forecast, anomaly, clustering
- `PowerBI_Superstore/`: chứa file dashboard Power BI
- `Report/`: chứa báo cáo và slide bảo vệ đồ án
- `streamlit_app.py`: file chính chạy ứng dụng web Streamlit
- `ai_analysis_superstore.py`: file xử lý, hỗ trợ phân tích AI/ML

## Công nghệ sử dụng
- Python
- Pandas, NumPy
- Matplotlib, Seaborn, Plotly
- Scikit-learn
- Streamlit
- Power BI

## Các chức năng chính
### 1. Dashboard Power BI
- Theo dõi KPI tổng quan
- Phân tích chi tiết theo khu vực, danh mục, sản phẩm
- Quan sát xu hướng doanh thu theo thời gian

### 2. Phân tích dữ liệu bằng Python
- Phân tích khám phá dữ liệu (EDA)
- Kiểm tra mối quan hệ giữa doanh thu, lợi nhuận, chiết khấu
- Khai thác insight phục vụ phân tích bán hàng

### 3. Dự báo doanh thu
- Xây dựng mô hình Linear Regression
- Đánh giá mô hình bằng MAE, RMSE, R²

### 4. Phát hiện bất thường
- Sử dụng Isolation Forest
- Nhận diện các giao dịch có dấu hiệu bất thường trong dữ liệu bán hàng

### 5. Phân cụm khách hàng
- Sử dụng K-Means
- Phân nhóm khách hàng theo đặc điểm giao dịch

### 6. Ứng dụng web Streamlit
- Hiển thị KPI và biểu đồ
- Tích hợp bộ lọc dữ liệu tương tác
- Hiển thị kết quả dự báo, phát hiện bất thường và phân cụm khách hàng
