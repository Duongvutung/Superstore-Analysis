# Dự án Phân tích Dữ liệu Bán lẻ Superstore (Superstore Sales Analytics & ML Pipeline)

> Dự án Portfolio Phân tích Dữ liệu  
> Kết hợp phân tích Khám phá (EDA), Dashboard tương tác (Power BI & Streamlit) và Học máy cơ bản (Forecasting, Anomaly Detection, Clustering) để giải quyết bài toán tối ưu hóa doanh thu và biên lợi nhuận chuỗi bán lẻ.

---

## 1. Tổng quan Dự án (Executive Summary)

* **Bộ dữ liệu:** Gồm 9.994 dòng giao dịch (tương ứng với 5.009 đơn hàng `Order ID`) của chuỗi bán lẻ Superstore trong giai đoạn 2014 – 2017.
* **Mục tiêu:**
  * Theo dõi các chỉ số KPI bán hàng cốt lõi (Doanh thu, Lợi nhuận, Số lượng, Đơn hàng).
  * Phát hiện nghịch lý biên lợi nhuận giữa các ngành hàng và rủi ro từ chính sách giảm giá.
  * Xây dựng mô hình dự báo doanh thu chuỗi thời gian nắm bắt tính mùa vụ.
  * Phân tầng giao dịch dị biệt (Anomaly) và phân khúc khách hàng theo giá trị để đưa ra khuyến nghị kinh doanh thực tế.

---

## 2. Những Phát hiện Kinh doanh Cốt lõi (Key Insights)

1. **Nghịch lý ngành hàng Nội thất (Furniture vs. Technology):**
   * Technology dẫn đầu về hiệu quả dòng tiền: Doanh thu 836.154 USD, mang lại 145.454 USD lợi nhuận (biên lợi nhuận ~17.4%).
   * Furniture đóng góp gần 1/3 doanh số (741.999 USD) nhưng lợi nhuận chỉ vỏn vẹn 18.451 USD (biên lợi nhuận cực mỏng ~2.49%), chủ yếu do các nhóm hàng cồng kềnh như Tables và Bookcases bị bán lỗ khi chiết khấu sâu.
2. **Bẫy chiết khấu (Discount Trap):**
   * Tương quan giữa `Discount` và `Profit` mang giá trị âm rõ rệt. Khi tỷ lệ chiết khấu vượt ngưỡng an toàn (>20% - 30%), xác suất đơn hàng bị âm vốn tăng đột biến.
3. **Cải tiến Mô hình Dự báo Doanh thu (Forecasting):**
   * Mô hình Hồi quy tuyến tính chỉ dựa vào trục thời gian đơn thuần (`TimeIndex`) cho kết quả kém ($R^2 \approx 0.034$) do không bắt được đỉnh bán hàng cuối năm (tháng 11, 12).
   * Khi tích hợp thêm các biến giả mùa vụ theo từng tháng (`Month One-Hot Encoding`), mô hình Hồi quy tuyến tính đa biến đã bám sát chu kỳ lặp lại hàng năm, nâng $R^2$ lên ~63% và giảm mạnh sai số dự báo so với Baseline.
4. **Phân tích Nâng cao (Outliers & Customer Segmentation):**
   * Giao dịch dị biệt (Isolation Forest): Thiết lập tham số gắn cờ cố định 3% dòng giao dịch rủi ro nhất để kiểm soát nội bộ, phát hiện các đơn hàng chiết khấu cực sâu gây lỗ nặng.
   * Phân cụm khách hàng (K-Means): Phân nhóm khách hàng thành các tầng giá trị rõ ràng (*High Value*, *Medium Value*, *Low Value*, *Emerging*) gắn liền với chiến lược chăm sóc và hạn mức ưu đãi riêng biệt.

*(Chi tiết câu chuyện dữ liệu và đề xuất hành động cụ thể xem tại file: [INSIGHTS_AND_STORYTELLING.md](INSIGHTS_AND_STORYTELLING.md))*

---

## 3. Công nghệ & Thư viện Sử dụng

* Ngôn ngữ & Thư viện cốt lõi: Python (Pandas, NumPy, Scikit-learn, Joblib)
* Trực quan hóa: Matplotlib, Seaborn, Plotly
* Dashboard & Ứng dụng: Power BI (`.pbix`), Streamlit (`streamlit_app.py`)
* Môi trường & Quản lý phiên bản: Git, GitHub, Conda / Virtualenv

---

## 📁 4. Cấu trúc Thư mục Dự án

```text
Superstore-Analysis/
├── Data/
│   └── Superstore Sale Dataset.csv      # Dữ liệu gốc bán hàng
├── Notebooks/
│   ├── Superstore_EDA.ipynb             # Phân tích khám phá & trực quan thống kê
│   ├── Sales_Forecasting.ipynb          # Pipeline dự báo doanh thu chuỗi thời gian
│   └── sales_forecasting_bundle.joblib  # Bundle model hồi quy mùa vụ đã huấn luyện
├── Outputs/                             # Bảng tổng hợp kết quả phân tích & ML
├── PowerBI_Superstore/                  # Dashboard báo cáo Power BI (.pbix)
├── ai_analysis_superstore.py            # Script chạy phân tích Isolation Forest & K-Means
├── streamlit_app.py                     # Web App Dashboard trực quan tương tác
├── requirements.txt                     # Danh sách thư viện và phiên bản môi trường
├── INSIGHTS_AND_STORYTELLING.md         # Báo cáo chi tiết Data Storytelling & Business Actions
└── README.md                            # Tổng quan dự án