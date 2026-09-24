# BÁO CÁO KHAI THÁC INSIGHT & DATA STORYTELLING: DỰ ÁN BÁN HÀNG SUPERSTORE

---

## 1. Bối cảnh & Câu chuyện dữ liệu (The Big Picture)

Trong giai đoạn 2014 – 2017, chuỗi bán lẻ Superstore ghi nhận tổng cộng **9.994 giao dịch**. Nhìn bề ngoài, doanh nghiệp liên tục tăng trưởng doanh thu qua từng năm. Tuy nhiên, khi đi sâu vào phân tích tương quan đa chiều, câu chuyện kinh doanh bộc lộ **3 nghịch lý lớn** đe dọa trực tiếp đến sự tồn vong và biên lợi nhuận của doanh nghiệp:

* **Bán nhiều nhưng không có lời:** Một số ngành hàng đóng góp doanh thu khổng lồ nhưng mang lại biên lợi nhuận mỏng manh.
* **Chiết khấu không kiểm soát (Discount vs Profit Trap):** Chính sách khuyến mãi giảm giá sâu để kích cầu đang trực tiếp tạo ra các khoản lỗ nặng nề.
* **Hiệu ứng "Chảy máu dòng tiền" từ khách hàng và sản phẩm dị biệt:** Một lượng nhỏ các giao dịch và khách hàng đang âm thầm bào mòn thành quả kinh doanh của toàn bộ chuỗi.

---

## 2. Các Insight Cốt lõi (Key Findings)

## 2. Các Insight Cốt lõi (Key Findings)

### 📌 Insight 1: Nghịch lý ngành hàng Nội thất (Furniture vs Technology)
* **Số liệu thực tế:** 
  * **Technology (Công nghệ):** Dẫn đầu về cả doanh thu và lợi nhuận, mang về 836.154,03 USD doanh thu và 145.454,95 USD lợi nhuận (biên lợi nhuận đạt ~17.4%).
  * **Furniture (Nội thất):** Tạo ra 741.999,80 USD doanh thu (chiếm gần một phần ba tổng doanh thu toàn chuỗi), nhưng lợi nhuận chỉ vỏn vẹn 18.451,27 USD (biên lợi nhuận cực kỳ mỏng, chỉ ~2.49%).
  * **Chi tiết nhóm hàng:** Trong khi *Phones* và *Chairs* cùng đạt mốc doanh thu rất cao (~328.000 USD - 330.000 USD), nhóm hàng *Tables* và *Bookcases* (thuộc Furniture) lại thường xuyên rơi vào tình trạng âm lợi nhuận do chi phí vận chuyển cồng kềnh và áp lực chiết khấu lớn.

### 📌 Insight 2: Bẫy chiết khấu (The Discount - Profit Trade-off)
* **Số liệu thực tế:** Hệ số tương quan giữa `Discount` và `Profit` mang giá trị âm rõ rệt.
* **Hiện tượng:** Khi tỷ lệ chiết khấu vượt ngưỡng an toàn (trên 20% - 30%), xác suất đơn hàng bị âm lợi nhuận tăng vọt. Nhiều đơn hàng có giá trị lớn nhưng sau khi áp dụng chiết khấu sâu (40% - 80%) đã tạo ra các khoản lỗ đột biến (từ -300 USD đến hàng nghìn USD).
* **Ý nghĩa:** Chính sách giảm giá hiện tại đang bị lạm dụng để chạy doanh số ảo thay vì bảo vệ lợi nhuận gộp.

### 📌 Insight 3: Động lực tăng trưởng vùng miền
* **Số liệu thực tế:**
  * Khu vực **West (miền Tây)** là đầu tàu kinh doanh với doanh thu đạt 725.457,82 USD.
  * Khu vực **East (miền Đông)** xếp thứ hai với 678.781,24 USD.
  * Hai khu vực **Central** và **South** có quy mô khiêm tốn hơn (khoảng 501.000 USD và 391.000 USD).
* **Ý nghĩa:** Nguồn lực marketing và mở rộng kho vận cần tập trung tối ưu cho 2 thị trường trọng điểm bờ Đông và bờ Tây.

### 📌 Insight 4: Tính mùa vụ và Mô hình dự báo
* **Số liệu thực tế:** Doanh thu biến động theo chu kỳ rõ rệt: chạm đáy vào Quý 1 (tháng 1, 2) và bùng nổ đạt đỉnh vào Quý 4 (tháng 11, 12 hàng năm).
* **Cải tiến:** Sau khi tích hợp thêm các biến giả mùa vụ theo từng tháng (`Month One-Hot Encoding`), mô hình hồi quy tuyến tính đa biến đã nắm bắt được quy luật tăng trưởng cuối năm, nâng hệ số R² lên mức xấp xỉ 60% và vượt trội so với mức dự báo cơ sở (Baseline).
* **Cải tiến:** Sau khi tích hợp thêm các biến giả mùa vụ theo từng tháng (`Month One-Hot Encoding`), mô hình hồi quy tuyến tính đa biến đã nắm bắt được quy luật tăng trưởng cuối năm, nâng hệ số $R^2$ lên mức xấp xỉ **60%** và vượt trội so với mức dự báo cơ sở (Baseline).

### 📌 Insight 5: Giao dịch dị biệt & Phân cụm khách hàng
* **Giao dịch dị biệt (Isolation Forest):** Khoảng 3% tổng số đơn hàng được xếp vào nhóm bất thường. Phần lớn các đơn hàng này có mức chiết khấu cực đại ($\ge 50\%$) trực tiếp gây thâm hụt dòng tiền nặng nề.
* **Phân cụm khách hàng (K-Means):**
  * **Nhóm VIP / Champions:** Chiếm tỷ lệ nhỏ nhưng đóng góp phần lớn doanh số và lợi nhuận ổn định.
  * **Nhóm At Risk / Price Sensitive:** Khách hàng chỉ mua khi có giảm giá sâu, biên lợi nhuận tạo ra rất thấp.

---

## 3. Hành động kinh doanh khuyến nghị (Actionable Recommendations)

1. **Quản trị danh mục và định giá lại ngành Nội thất (Furniture):**
   * Rà soát lại chi phí logistics đối với nhóm *Tables* và *Bookcases*.
   * Hạn chế bán đơn lẻ nhóm hàng này với chiết khấu cao; đóng gói (bundle) bán kèm với các phụ kiện công nghệ (*Technology*) có biên lợi nhuận dày để bù đắp thâm hụt.
2. **Thiết lập rào chắn chiết khấu (Discount Guardrails):**
   * Đặt trần chiết khấu tự động trên phần mềm bán hàng: nhân viên chỉ được giảm tối đa **15% – 20%**.
   * Mức giảm giá từ **30% trở lên** bắt buộc phải qua phê duyệt của Quản lý kinh doanh.
3. **Cá nhân hóa chăm sóc khách hàng theo cụm (Customer Clustering Strategy):**
   * **Nhóm Champions/VIP:** Xây dựng chương trình khách hàng thân thiết, dịch vụ hỗ trợ ưu tiên để giữ chân.
   * **Nhóm At Risk / Giá rẻ:** Chuyển đổi chiến lược từ giảm giá trực tiếp sang tặng quà phụ kiện hoặc điểm tích lũy.
4. **Vận hành kế hoạch theo mùa vụ (Seasonal Inventory Planning):**
   * Chủ động nhập hàng và tăng năng lực kho bãi trước tháng 10 hàng năm để đáp ứng đợt bùng nổ doanh số tháng 11, 12.