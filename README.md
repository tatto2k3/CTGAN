# Fraud Detection với CTGAN Data Augmentation
## Dataset: JAR2020 – Bao et al. (2020)

## Cấu trúc project
```
fraud_ctgan/
├── data/                          # Đặt file data_FraudDetection_JAR2020.csv vào đây
├── src/
│   ├── 00_feature_analysis.py     # Phân tích nên dùng raw/ratio/cả hai
│   ├── 01_eda.py                  # Chương III.2 – EDA
│   ├── 02_preprocessing.py        # Chương III.3 – Tiền xử lý
│   ├── 03_ctgan_train.py          # Chương III.4 – Huấn luyện CTGAN
│   ├── 04_experiment.py           # Chương III.5 + IV – Thực nghiệm
│   └── utils.py                   # Hàm dùng chung
├── outputs/                       # Kết quả, biểu đồ, model
└── README.md
```

## Cách chạy
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost ctgan imbalanced-learn scipy

# Bước 0: Phân tích feature (quan trọng!)
python src/00_feature_analysis.py

# Bước 1: EDA
python src/01_eda.py

# Bước 2: Tiền xử lý
python src/02_preprocessing.py

# Bước 3: Huấn luyện CTGAN
python src/03_ctgan_train.py

# Bước 4: Thực nghiệm so sánh
python src/04_experiment.py
```
