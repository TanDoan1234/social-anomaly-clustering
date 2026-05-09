<p align="center">
  <img src="./assets/social_anomaly_clustering_animated_logo.gif" alt="Social Anomaly Clustering Logo" width="150">
</p>

# GOM NHÓM PHÁT HIỆN BẤT THƯỜNG - ĐỒ ÁN CUỐI KỲ KHAI PHÁ DỮ LIỆU

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib" />
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter" />
  <img src="https://img.shields.io/badge/YAML-CB171E?style=for-the-badge&logo=yaml&logoColor=white" alt="YAML" />
</p>

<div align="center">
  <b><a href="./README.md">English Version</a></b>
</div>

---

## Thông tin sinh viên

<p align="center">
  <a href="https://huit.edu.vn/">
    <img src="./assets/Logo%20HUIT-03.png" alt="HUIT Logo" width="300">
  </a>
</p>

| MSSV | Họ và tên | GitHub | Email |
|:----------:|------------------|-----------------------------------------|------------------------|
| 2001230791 | Đoàn Tấn Minh Tân | [TanDoan1234](https://github.com/TanDoan1234) | doanminhtan.dev@gmail.com |

---

## 1. Tổng quan dự án

**Social Anomaly Clustering** là một dự án khai phá dữ liệu áp dụng các kỹ thuật học máy không giám sát để phát hiện các hành vi bất thường của người dùng trên các nền tảng mạng xã hội, cụ thể là Twitter/X.

Dự án không nhằm mục đích xây dựng một bộ phân loại bot có giám sát. Thay vào đó, nó tập trung vào việc khám phá các mẫu hành vi ẩn từ siêu dữ liệu người dùng và các đặc trưng liên quan đến tương tác. Những người dùng có hành vi sai lệch mạnh so với các nhóm hành vi chính sẽ được coi là người dùng bất thường tiềm năng.

Dự án so sánh hai phương pháp tiếp cận dựa trên phân cụm:

- **Phân cụm K-Means**
- **Phân cụm DBSCAN**

Cột `bot_label` có sẵn không được sử dụng trong quá trình phân cụm. Nó chỉ được sử dụng sau khi phân cụm để đánh giá liệu các nhóm bất thường được phát hiện có chứa tỷ lệ tài khoản bot cao hơn hay không.

---

## 2. Đặt vấn đề

Các nền tảng mạng xã hội thường chứa nhiều loại người dùng bất thường khác nhau, bao gồm các tài khoản spam, tài khoản giống bot và người dùng có mẫu tương tác không bình thường. Các tài khoản này có xu hướng hành xử khác với đa số người dùng bình thường.

Tuy nhiên, trong nhiều kịch bản thực tế, các nhãn dữ liệu tin cậy có thể không có sẵn hoặc tốn kém để thu thập. Do đó, các phương pháp học không giám sát rất hữu ích để khám phá các hành vi nghi ngờ mà không cần dựa trực tiếp vào dữ liệu đã dán nhãn.

Dự án này giải quyết câu hỏi sau:

> Liệu các thuật toán phân cụm có thể xác định được các nhóm người dùng có hành vi sai lệch so với đa số người dùng mạng xã hội hay không?

Ý tưởng chính là:

```text
Người dùng bình thường có xu hướng tạo thành các nhóm hành vi dày đặc hoặc ổn định.
Người dùng bất thường có xu hướng xuất hiện xa các cụm chính hoặc ở các vùng có mật độ thấp.
```

---

## 3. Mô tả bộ dữ liệu

Dự án sử dụng bộ dữ liệu **Twitter Bot Detection Dataset** từ Kaggle.

* **Nguồn dữ liệu:** [Twitter Bot Detection Dataset - Kaggle](https://www.kaggle.com/datasets/goyaladi/twitter-bot-detection-dataset?select=bot_detection_data.csv)
* **File đầu vào:** `bot_detection_data.csv`
* **Số lượng bản ghi:** 50,000
* **Số lượng cột gốc:** 11

Dữ liệu thô được đặt tại:

```text
datasets/raw/bot_detection_data.csv
```

### 3.1 Các cột dữ liệu thô

| Cột gốc          | Cột đã đổi tên    | Mô tả                                 | Sử dụng để phân cụm              |
| ---------------- | ---------------- | ------------------------------------- | -------------------------------- |
| `User ID`        | `user_id`        | Mã định danh duy nhất của người dùng  | Không                            |
| `Username`       | `username`       | Tên người dùng Twitter/X              | Dùng để tạo `username_length`    |
| `Tweet`          | `tweet`          | Nội dung văn bản của Tweet            | Dùng để tạo `tweet_length`       |
| `Retweet Count`  | `retweet_count`  | Số lượng retweet                      | Có                               |
| `Mention Count`  | `mention_count`  | Số lượng nhắc tên (mention)           | Có                               |
| `Follower Count` | `follower_count` | Số lượng người theo dõi               | Có                               |
| `Verified`       | `verified`       | Trạng thái xác thực tài khoản         | Có, chuyển đổi sang 0/1          |
| `Bot Label`      | `bot_label`      | Nhãn Người/Bot                        | Không, chỉ dùng để đánh giá      |
| `Location`       | `location`       | Vị trí do người dùng khai báo         | Không dùng trong đặc trưng cuối  |
| `Created At`     | `created_at`     | Dấu thời gian tạo tài khoản/bản ghi   | Dùng để tạo `account_age_days`   |
| `Hashtags`       | `hashtags`       | Các hashtag trong tweet               | Dùng để tạo các đặc trưng hashtag|

### 3.2 Cách sử dụng nhãn

Bộ dữ liệu chứa cột `bot_label`. Tuy nhiên, cột này không được sử dụng làm đặc trưng huấn luyện vì dự án tuân theo phương pháp khai phá dữ liệu không giám sát.

Vai trò của `bot_label` là:

```text
Sử dụng trong phân cụm: Không
Sử dụng trong đánh giá: Có
```

Điều này đảm bảo rằng quá trình phân cụm chỉ dựa trên các đặc trưng hành vi của người dùng.

---

## 4. Cấu trúc dự án

```text
social-anomaly-clustering/
│
├── README.md
├── README_VI.md
├── requirements.txt
├── main.py
│
├── configs/
│   └── config.yaml
│
├── assets/
│   ├── social_anomaly_clustering_animated_logo.gif
│   └── Logo HUIT-03.png
│
├── datasets/
│   ├── raw/
│   │   └── bot_detection_data.csv
│   │
│   ├── cleaned/
│   │   └── bot_detection_clean.csv
│   │
│   └── processed/
│       ├── user_features.csv
│       ├── user_features_scaled.csv
│       └── user_features_with_label.csv
│
├── notebooks/
│   ├── 01_explore_dataset.ipynb
│   ├── 02_preprocessing_feature_engineering.ipynb
│   ├── 03_kmeans_clustering.ipynb
│   ├── 04_dbscan_clustering.ipynb
│   └── 05_evaluation_visualization.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── clustering.py
│   ├── evaluation.py
│   └── visualization.py
│
└── results/
    ├── models/
    │   ├── scaler.pkl
    │   ├── kmeans_model.pkl
    │   ├── dbscan_model.pkl
    │   └── pca_model.pkl
    │
    ├── csv/
    │   ├── kmeans_clustered_users.csv
    │   ├── kmeans_anomaly_users.csv
    │   ├── dbscan_clustered_users.csv
    │   ├── dbscan_anomaly_users.csv
    │   ├── cluster_summary.csv
    │   ├── dbscan_cluster_summary.csv
    │   ├── evaluation_summary.csv
    │   └── algorithm_comparison.csv
    │
    ├── figures/
    │   ├── label_distribution.png
    │   ├── correlation_matrix.png
    │   ├── elbow_method.png
    │   ├── silhouette_score.png
    │   ├── pca_kmeans.png
    │   ├── pca_kmeans_anomaly.png
    │   ├── pca_dbscan.png
    │   ├── pca_dbscan_anomaly.png
    │   ├── compare_anomaly_users.png
    │   ├── compare_bot_ratio_normal_vs_anomaly.png
    │   └── compare_silhouette_score.png
    │
    └── logs/
        └── experiment.log
```

---

## 5. Quy trình xử lý dữ liệu (Data Pipeline)

Quy trình làm việc hoàn chỉnh được trình bày dưới đây.

<p align="center">
  <img src="./assets/pipeline.png" alt="Quy trình xử lý dữ liệu" width="800">
</p>

### Lưu ý quan trọng về PCA

PCA chỉ được sử dụng cho mục đích trực quan hóa.
Các mô hình phân cụm được huấn luyện trên tập đặc trưng đã chuẩn hóa đầy đủ, không phải trên đầu ra PCA 2 chiều.

---

## 6. Luồng công việc của Notebook

| Notebook                                     | Mục đích chính                                                               | Kết quả đầu ra chính                                     |
| -------------------------------------------- | ---------------------------------------------------------------------------- | -------------------------------------------------------- |
| `01_explore_dataset.ipynb`                   | Khám phá cấu trúc dữ liệu, giá trị khuyết thiếu, phân phối, cân bằng nhãn     | Các biểu đồ EDA và quan sát ban đầu                      |
| `02_preprocessing_feature_engineering.ipynb` | Làm sạch dữ liệu, đổi tên cột, xử lý giá trị khuyết, tạo đặc trưng số         | `user_features.csv`, `user_features_scaled.csv`          |
| `03_kmeans_clustering.ipynb`                 | Áp dụng K-Means, chọn K, tính toán điểm bất thường                           | `kmeans_clustered_users.csv`, `kmeans_anomaly_users.csv` |
| `04_dbscan_clustering.ipynb`                 | Áp dụng DBSCAN, điều chỉnh `eps` và `min_samples`, phát hiện nhiễu           | `dbscan_clustered_users.csv`, `dbscan_anomaly_users.csv` |
| `05_evaluation_visualization.ipynb`          | So sánh kết quả K-Means và DBSCAN                                            | `evaluation_summary.csv`, các biểu đồ so sánh            |

---

## 7. Làm sạch và Tiền xử lý dữ liệu

### 7.1 Đổi tên cột

Các tên cột ban đầu chứa khoảng trắng và chữ hoa. Chúng đã được chuyển đổi sang định dạng `snake_case`.

Ví dụ:

```text
User ID        -> user_id
Retweet Count -> retweet_count
Mention Count -> mention_count
Follower Count -> follower_count
Bot Label     -> bot_label
Created At    -> created_at
```

### 7.2 Xử lý giá trị khuyết thiếu

Cột `hashtags` chứa các giá trị khuyết thiếu. Những giá trị này được thay thế bằng một chuỗi rỗng:

```python
df_clean["hashtags"] = df_clean["hashtags"].fillna("")
```

Điều này là hợp lý vì trường hashtag bị thiếu có thể được hiểu là một tweet không có hashtag.

### 7.3 Chuyển đổi kiểu dữ liệu

Các chuyển đổi sau đã được thực hiện:

| Cột          | Trước         | Sau         |
| ------------ | ------------- | ----------- |
| `verified`   | Kiểu Boolean  | Số nguyên 0/1 |
| `created_at` | Chuỗi/Object  | Kiểu Datetime |

Cột `created_at` được chuyển sang datetime để có thể tính toán tuổi đời tài khoản.

---

## 8. Feature Engineering

Bộ dữ liệu gốc chứa các trường văn bản, phân loại, số và thời gian. Vì K-Means và DBSCAN yêu cầu đầu vào là số, một số đặc trưng số đã được tạo ra.

### 8.1 Các đặc trưng cuối cùng được dùng để phân cụm

| Đặc trưng          | Mô tả                                                              |
| ------------------ | ------------------------------------------------------------------ |
| `retweet_count`    | Số lượng retweet                                                   |
| `mention_count`    | Số lượng nhắc tên                                                  |
| `follower_count`   | Số lượng người theo dõi                                            |
| `verified`         | Trạng thái xác thực chuyển đổi sang 0/1                            |
| `tweet_length`     | Số lượng ký tự trong tweet                                         |
| `username_length`  | Số lượng ký tự trong tên người dùng                                |
| `hashtag_count`    | Số lượng hashtag được sử dụng                                      |
| `has_hashtag`      | Đặc trưng nhị phân cho biết có hashtag hay không                   |
| `account_age_days` | Số ngày từ `created_at` đến ngày mới nhất trong bộ dữ liệu         |

### 8.2 Các cột bị loại bỏ

| Cột          | Lý do loại bỏ                                    |
| ------------ | ------------------------------------------------ |
| `user_id`    | Chỉ là mã định danh, không mang thông tin hành vi|
| `username`   | Trường văn bản, đã chuyển sang `username_length` |
| `tweet`      | Trường văn bản, đã chuyển sang `tweet_length`    |
| `location`   | Không sử dụng trong tập đặc trưng cuối cùng      |
| `created_at` | Đã chuyển đổi thành `account_age_days`           |
| `hashtags`   | Đã chuyển thành `hashtag_count` và `has_hashtag` |
| `bot_label`  | Chỉ dùng để đánh giá, không dùng để phân cụm     |

---

## 9. Chuẩn hóa dữ liệu (Feature Scaling)

Vì các đặc trưng được chọn có phạm vi giá trị khác nhau, việc chuẩn hóa dữ liệu là bắt buộc trước khi áp dụng phân cụm.

Ví dụ:

```text
follower_count      từ 0 đến 10000
mention_count       từ 0 đến 5
verified            từ 0 đến 1
account_age_days    tính bằng ngày
```

Nếu không chuẩn hóa, các đặc trưng có giá trị lớn như `follower_count` và `account_age_days` có thể lấn át các đặc trưng khác trong phân cụm dựa trên khoảng cách.

Dự án này sử dụng:

```text
StandardScaler
```

StandardScaler biến đổi mỗi đặc trưng để có:

```text
trung bình (mean) = 0
độ lệch chuẩn (standard deviation) = 1
```

Bộ dữ liệu đã chuẩn hóa được lưu tại:

```text
datasets/processed/user_features_scaled.csv
```

Scaler đã khớp được lưu tại:

```text
results/models/scaler.pkl
```

---

## 10. Thuật toán gom nhóm

## 10.1 Phân cụm K-Means

K-Means là thuật toán phân cụm dựa trên tâm điểm. Nó nhóm các điểm dữ liệu bằng cách cực tiểu hóa khoảng cách giữa mỗi điểm và tâm cụm gần nhất.

### Lựa chọn K

Dự án đã đánh giá các giá trị `K` khác nhau bằng:

* Phương pháp Elbow
* Chỉ số Silhouette

Chỉ số Silhouette cao nhất tại `K = 2`. Tuy nhiên, **`K = 3`** đã được chọn cho thực nghiệm cuối cùng vì nó cung cấp các nhóm hành vi dễ giải thích hơn cho việc phân tích bất thường.

### Phát hiện bất thường bằng K-Means

K-Means không trực tiếp đưa ra các điểm bất thường. Do đó, việc phát hiện bất thường được thực hiện bằng cách tính khoảng cách đến tâm cụm gần nhất:

```text
anomaly_score = khoảng cách từ người dùng đến tâm cụm gần nhất
```

Top 5% người dùng có điểm bất thường cao nhất được đánh dấu là người dùng bất thường.

Thiết lập K-Means cuối cùng:

```text
n_clusters = 3
ngưỡng bất thường = top 5% điểm anomaly_score cao nhất
```

---

## 10.2 Phân cụm DBSCAN

DBSCAN là thuật toán phân cụm dựa trên mật độ. Nó nhóm các điểm nằm gần nhau trong các vùng dày đặc và đánh dấu các điểm cô lập là nhiễu.

Khác với K-Means, DBSCAN không yêu cầu biết trước số lượng cụm.

### Các tham số DBSCAN

Các tham số chính bao gồm:

| Tham số       | Ý nghĩa                                                           |
| ------------- | ----------------------------------------------------------------- |
| `eps`         | Khoảng cách tối đa giữa các điểm lân cận                          |
| `min_samples` | Số lượng điểm lân cận tối thiểu để tạo thành một vùng dày đặc    |

Sau khi thử nghiệm nhiều tổ hợp tham số, cấu hình được chọn là:

```text
eps = 1.1
min_samples = 15
```

Cấu hình này tạo ra tỷ lệ bất thường hợp lý và cấu trúc cụm ổn định.

### Phát hiện bất thường bằng DBSCAN

Trong DBSCAN, các điểm được gán vào cụm `-1` được coi là nhiễu.

```text
cụm = -1 -> người dùng bất thường
```

---

## 11. Kết quả thực nghiệm

### 11.1 So sánh tổng thể

| Thuật toán  | Số lượng cụm | Người dùng bất thường | Tỷ lệ bất thường | Chỉ số Silhouette |
| ----------- | -----------: | --------------------: | ---------------: | ----------------: |
| **K-Means** |            3 |                 2,500 |            5.00% |             0.151 |
| **DBSCAN**  |            4 |                 2,816 |            5.63% |             0.123 |

### Diễn giải

K-Means phát hiện chính xác 2,500 người dùng bất thường vì ngưỡng được thiết lập thủ công là top 5% điểm bất thường.

DBSCAN tự động phát hiện 2,816 người dùng bất thường dưới dạng các điểm nhiễu, tương ứng với 5.63% bộ dữ liệu.

K-Means đạt chỉ số Silhouette cao hơn DBSCAN, cho thấy K-Means tạo ra các cụm được phân tách tốt hơn một chút trong thực nghiệm này.

---

## 11.2 Đánh giá tỷ lệ Bot

| Thuật toán  | Tỷ lệ Bot bình thường | Tỷ lệ Bot bất thường | Chênh lệch |
| ----------- | --------------------: | --------------------: | ---------: |
| **K-Means** |                49.95% |                51.64% |     +1.69% |
| **DBSCAN**  |                50.02% |                50.28% |     +0.26% |

### Diễn giải

Nhóm bất thường được phát hiện bởi K-Means có tỷ lệ bot cao hơn nhóm bình thường là 1.69%.

Nhóm bất thường được phát hiện bởi DBSCAN cũng có tỷ lệ bot cao hơn một chút so với nhóm bình thường, nhưng chênh lệch chỉ là 0.26%.

Điều này cho thấy K-Means đã bắt được các người dùng bất thường có liên quan nhiều hơn một chút đến hành vi giống bot trong không gian đặc trưng này. Tuy nhiên, sự khác biệt vẫn tương đối nhỏ, nghĩa là các đặc trưng hiện tại chưa đủ mạnh để phân tách rõ ràng tài khoản bot và người thật.

---

## 11.3 So sánh đặc trưng giữa nhóm Bình thường và Bất thường

| Thuật toán| Nhóm      | Retweet Count | Mention Count | Follower Count | Tweet Length | Username Length | Hashtag Count |
| --------- | --------- | ------------: | ------------: | -------------: | -----------: | --------------: | ------------: |
| K-Means   | Bình thường|         49.94 |          2.51 |        4990.60 |        62.43 |            9.69 |          2.50 |
| K-Means   | Bất thường |         51.20 |          2.53 |        4950.64 |        66.30 |           11.92 |          2.45 |
| DBSCAN    | Bình thường|         49.96 |          2.51 |        4992.97 |        62.23 |            9.60 |          2.54 |
| DBSCAN    | Bất thường |         50.76 |          2.53 |        4915.41 |        69.21 |           13.14 |          1.84 |

### Diễn giải

Đối với K-Means, người dùng bất thường có xu hướng có tweet dài hơn và tên người dùng dài hơn so với người dùng bình thường.

Đối với DBSCAN, người dùng bất thường cũng cho thấy độ dài tweet và độ dài tên người dùng cao hơn, nhưng sử dụng hashtag ít hơn so với người dùng bình thường.

Điều này cho thấy hành vi bất thường trong thực nghiệm này không chỉ liên quan đến các chỉ số tương tác như retweet và nhắc tên, mà còn liên quan đến các mẫu văn bản và siêu dữ liệu tài khoản.

---

## 12. Trực quan hóa kết quả

### 12.1 Phân tích dữ liệu khám phá (EDA)

<table align="center">
  <tr>
    <td align="center">
      <img src="./results/figures/correlation_matrix.png" width="400px"/>
      <br/>
      <b>Ma trận tương quan đặc trưng</b>
    </td>
    <td align="center">
      <img src="./results/figures/label_distribution.png" width="400px"/>
      <br/>
      <b>Phân phối Bot vs Người thật</b>
    </td>
  </tr>
</table>

### 12.2 Lựa chọn mô hình

<table align="center">
  <tr>
    <td align="center">
      <img src="./results/figures/elbow_method.png" width="400px"/>
      <br/>
      <b>Phương pháp Elbow cho K-Means</b>
    </td>
    <td align="center">
      <img src="./results/figures/silhouette_score.png" width="400px"/>
      <br/>
      <b>Chỉ số Silhouette cho K-Means</b>
    </td>
  </tr>
</table>

### 12.3 Trực quan hóa Phân cụm và Bất thường

<table align="center">
  <tr>
    <td align="center">
      <img src="./results/figures/pca_kmeans_anomaly.png" width="400px"/>
      <br/>
      <b>Phát hiện bất thường K-Means bằng PCA</b>
    </td>
    <td align="center">
      <img src="./results/figures/pca_dbscan_anomaly.png" width="400px"/>
      <br/>
      <b>Phát hiện bất thường DBSCAN bằng PCA</b>
    </td>
  </tr>
</table>

### 12.4 So sánh thuật toán

<table align="center">
  <tr>
    <td align="center">
      <img src="./results/figures/compare_anomaly_users.png" width="400px"/>
      <br/>
      <b>Số lượng người dùng bất thường</b>
    </td>
    <td align="center">
      <img src="./results/figures/compare_bot_ratio_normal_vs_anomaly.png" width="400px"/>
      <br/>
      <b>Tỷ lệ Bot: Bình thường vs Bất thường</b>
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2">
      <img src="./results/figures/compare_silhouette_score.png" width="400px"/>
      <br/>
      <b>So sánh chỉ số Silhouette</b>
    </td>
  </tr>
</table>

---

## 13. Hạn chế của dự án

Mặc dù dự án đã áp dụng thành công các phương pháp phân cụm để phát hiện người dùng bất thường, vẫn còn một số hạn chế:

* Các đặc trưng được chọn tương đối đơn giản và có thể không phản ánh đầy đủ các hành vi bot phức tạp.
* Nội dung Tweet chỉ được đại diện bởi `tweet_length`, chưa sử dụng các nhúng văn bản (embeddings) mang ý nghĩa ngữ nghĩa.
* Mô hình không sử dụng các đặc trưng dựa trên đồ thị như cấu trúc người theo dõi/đang theo dõi.
* Vì phân cụm không học từ `bot_label`, người dùng bất thường có thể không khớp hoàn toàn với các tài khoản bot.
* Trực quan hóa PCA chỉ được dùng để diễn giải 2D và không đại diện cho toàn bộ không gian đặc trưng đa chiều.
* DBSCAN rất nhạy cảm với việc chọn tham số `eps` và `min_samples`.

---

## 14. Cách chạy dự án

### 14.1 Clone Repository

```bash
git clone https://github.com/TanDoan1234/social-anomaly-clustering.git
cd social-anomaly-clustering
```

### 14.2 Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 14.3 Chuẩn bị dữ liệu

Đặt file bộ dữ liệu tại:

```text
datasets/raw/bot_detection_data.csv
```

### 14.4 Chạy các Notebook

Mở Jupyter Notebook hoặc JupyterLab và chạy các notebook theo thứ tự:

```text
01_explore_dataset.ipynb
02_preprocessing_feature_engineering.ipynb
03_kmeans_clustering.ipynb
04_dbscan_clustering.ipynb
05_evaluation_visualization.ipynb
```

### 14.5 Kết quả đầu ra dự kiến

Sau khi chạy tất cả các notebook, các kết quả sau sẽ được tạo ra:

```text
datasets/cleaned/bot_detection_clean.csv
datasets/processed/user_features.csv
datasets/processed/user_features_scaled.csv
datasets/processed/user_features_with_label.csv

results/csv/kmeans_clustered_users.csv
results/csv/kmeans_anomaly_users.csv
results/csv/dbscan_clustered_users.csv
results/csv/dbscan_anomaly_users.csv
results/csv/evaluation_summary.csv

results/models/scaler.pkl
results/models/kmeans_model.pkl
results/models/dbscan_model.pkl

results/figures/elbow_method.png
results/figures/silhouette_score.png
results/figures/pca_kmeans_anomaly.png
results/figures/pca_dbscan_anomaly.png
results/figures/compare_anomaly_users.png
results/figures/compare_bot_ratio_normal_vs_anomaly.png
results/figures/compare_silhouette_score.png
```

---

## 15. Kết luận

Dự án này chứng minh cách các kỹ thuật phân cụm không giám sát có thể được áp dụng để phát hiện hành vi người dùng bất thường trên mạng xã hội.

K-Means và DBSCAN đều có thể xác định một nhóm nhỏ người dùng có hành vi khác biệt so với đa số. K-Means phát hiện 2,500 người dùng bất thường, trong khi DBSCAN phát hiện 2,816 người dùng bất thường.

K-Means đạt chỉ số Silhouette cao hơn và tạo ra nhóm bất thường có tỷ lệ bot cao hơn một chút so với DBSCAN. Tuy nhiên, cả hai phương pháp đều cho thấy tập đặc trưng hiện tại chưa đủ mạnh để phân biệt rõ ràng tài khoản bot và người thật.

Dự án đã thực hiện thành công một quy trình khai phá dữ liệu đầy đủ:

```text
Khám phá dữ liệu
-> Tiền xử lý dữ liệu
-> Kỹ nghệ đặc trưng
-> Chuẩn hóa dữ liệu
-> Phân cụm
-> Phát hiện bất thường
-> Đánh giá
-> Trực quan hóa
```

Nhìn chung, công trình này phù hợp làm đồ án cuối kỳ môn Khai phá dữ liệu và có thể phục vụ như một nền tảng cho các hệ thống phát hiện bất thường trên mạng xã hội nâng cao hơn.

---
