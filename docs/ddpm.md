# Hướng dẫn Cài đặt DDPM (Denoising Diffusion Probabilistic Models)

Tài liệu này tổng hợp các thành phần cốt lõi và công thức toán học cần thiết để lập trình một mô hình DDPM hoàn chỉnh (thường được triển khai bằng PyTorch).

---

## 1. Lịch trình Nhiễu (Noise Scheduler)
Module này quản lý quá trình thêm nhiễu (Forward Process) và cung cấp các hệ số cho quá trình khử nhiễu (Reverse Process). Bạn cần khởi tạo và tính toán trước các tensor sau:

* **Tổng số bước ($T$):** Thường thiết lập là 1000.
* **Variance Schedule ($\beta_t$):** Mảng các giá trị tăng dần tuyến tính từ $\beta_1$ (ví dụ: 0.0001) đến $\beta_T$ (ví dụ: 0.02).
* **Các hằng số phái sinh:**
    * $\alpha_t = 1 - \beta_t$
    * $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$ (Tích lũy của $\alpha_t$)
    * $\sqrt{\bar{\alpha}_t}$ và $\sqrt{1 - \bar{\alpha}_t}$ (Dùng cho công thức Forward Process).

---

## 2. Kiến trúc U-Net (Noise Predictor)
Mạng nơ-ron $\epsilon_\theta(x_t, t)$ có nhiệm vụ dự đoán lượng nhiễu đã được thêm vào ảnh tại bước $t$. 

* **Time Embedding:** * Sử dụng Sinusoidal Positional Embeddings để mã hóa timestep $t$ thành một vector.
    * Cộng hoặc nhân vector thời gian này vào các feature map trong U-Net để mạng nhận biết được mức độ nhiễu hiện tại.
* **Down-blocks (Mã hóa):** Gồm các lớp Conv2d, GroupNorm, SiLU (Swish) và các phép toán giảm kích thước không gian (MaxPooling hoặc Conv2d với stride=2).
* **Up-blocks (Giải mã):** Sử dụng ConvTranspose2d hoặc Upsample kết hợp Conv2d để khôi phục độ phân giải.
* **Skip Connections:** Nối (concatenate) feature map từ các Down-blocks sang các Up-blocks có cùng độ phân giải.

---

## 3. Quá trình Huấn luyện (Training - Forward Process)
Mục tiêu là dạy U-Net dự đoán chính xác lượng nhiễu $\epsilon$ đã được bơm vào ảnh.

**Các bước trong một vòng lặp (iteration):**
1.  Lấy một batch ảnh gốc $x_0$ và chuẩn hóa pixel về dải $[-1, 1]$.
2.  Lấy mẫu ngẫu nhiên timestep $t \sim \text{Uniform}(\{1, \dots, T\})$ cho mỗi ảnh trong batch.
3.  Lấy mẫu nhiễu ngẫu nhiên $\epsilon \sim \mathcal{N}(0, I)$.
4.  Tạo ảnh nhiễu $x_t$ bằng công thức:
    $$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$
5.  Dự đoán nhiễu thông qua U-Net: $\hat{\epsilon} = \epsilon_\theta(x_t, t)$.
6.  Tính hàm mục tiêu (MSE Loss) và cập nhật trọng số:
    $$L = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

---

## 4. Quá trình Lấy mẫu (Sampling - Reverse Process)
Sinh ảnh mới từ một tensor nhiễu trắng thuần túy.

**Thuật toán Ancestral Sampling:**
1.  Khởi tạo tensor $x_T \sim \mathcal{N}(0, I)$ với shape của ảnh mong muốn.
2.  Chạy vòng lặp ngược từ $t = T, T-1, \dots, 1$:
    * Dự đoán nhiễu: $\hat{\epsilon} = \epsilon_\theta(x_t, t)$.
    * Khởi tạo nhiễu ngẫu nhiên $z \sim \mathcal{N}(0, I)$. Nếu $t = 1$, thiết lập $z = 0$.
    * Khử nhiễu $x_t$ để tạo ra $x_{t-1}$ theo công thức:
        $$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \hat{\epsilon} \right) + \sqrt{\beta_t} z$$
3.  Kết thúc vòng lặp, un-normalize tensor $x_0$ từ $[-1, 1]$ về lại $[0, 1]$ để lưu hoặc hiển thị ảnh.

---

## 5. Lưu ý Kỹ thuật Xử lý Dữ liệu (Data Pipeline)
* **Chuẩn hóa:** Bắt buộc áp dụng biến đổi (transforms) để đưa toàn bộ giá trị pixel về dải $[-1, 1]$. Ví dụ với ảnh RGB: `transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])`. Việc này giúp dữ liệu đầu vào khớp với phân phối chuẩn có trung bình (mean) bằng 0 của nhiễu.