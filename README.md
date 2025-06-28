# 🔧 Sistem Prediksi Maintenance Pompa Canggih

Sistem machine learning komprehensif untuk prediksi kebutuhan maintenance pompa industri menggunakan ensemble model canggih, explainable AI, dan arsitektur produksi yang scalable.

## 📊 Rumus Matematika (DIKOREKSI)

### 1. Random Forest

Random Forest menggunakan ensemble dari decision trees:

```
Prediksi_RF = mode(Prediksi_Tree_1, Prediksi_Tree_2, ..., Prediksi_Tree_n)  [untuk klasifikasi]
Prediksi_RF = (1/n) * Σ(Prediksi_Tree_i) untuk i = 1 sampai n              [untuk regresi]
```

**Feature Importance (Mean Decrease Impurity):**

```
Importance(feature_j) = (1/n_trees) * Σ Σ p(t) * ΔI(t,j)
```

dimana:

- `p(t)` = proporsi sampel yang mencapai node t
- `ΔI(t,j)` = penurunan impurity saat split pada feature j di node t
- `n_trees` = jumlah pohon dalam forest

**Gini Impurity:**

```
Gini(t) = 1 - Σ p(i|t)²
```

### 2. XGBoost

Gradient Boosting dengan fungsi objektif:

```
Obj^(t) = Σ l(y_i, ŷ_i^(t-1) + f_t(x_i)) + Ω(f_t)
```

**Fungsi Loss (Log-loss untuk Binary Classification):**

```
l(y_i, ŷ_i) = y_i * log(p_i) + (1 - y_i) * log(1 - p_i)
dimana p_i = 1 / (1 + exp(-ŷ_i))
```

**Regularisasi:**

```
Ω(f_t) = γT + (1/2)λ Σ w_j²
```

dimana:

- `T` = jumlah leaves dalam pohon ke-t
- `w_j` = weight pada leaf j
- `γ` = minimum loss reduction untuk split
- `λ` = L2 regularization parameter

**Optimal Weight:**

```
w_j* = -G_j / (H_j + λ)
```

dimana:

- `G_j` = Σ g_i` (first-order gradient)
- `H_j` = Σ h_i` (second-order gradient)

### 3. LightGBM

Optimisasi pertumbuhan pohon leaf-wise dengan Gradient-based One-Side Sampling (GOSS):

**Split Gain:**

```
Gain = (1/2) * [G_L²/(H_L + λ) + G_R²/(H_R + λ) - (G_L + G_R)²/(H_L + H_R + λ)] - γ
```

**GOSS Sampling:**

```
Variance_j = (1/n) * Σ (g_i)² untuk feature j
```

**Exclusive Feature Bundling (EFB):**

```
Bundle features yang jarang konflik untuk menghemat memory
```

### 4. Metrik Evaluasi

**Akurasi:**

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Presisi:**

```
Precision = TP / (TP + FP)
```

**Recall (Sensitivity):**

```
Recall = TP / (TP + FN)
```

**Specificity:**

```
Specificity = TN / (TN + FP)
```

**F1-Score:**

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**F-beta Score:**

```
F_β = (1 + β²) * (Precision * Recall) / (β² * Precision + Recall)
```

**Area Under ROC Curve (AUC):**

```
AUC = ∫₀¹ TPR(FPR⁻¹(t)) dt
dimana TPR = TP/(TP+FN), FPR = FP/(FP+TN)
```

**Matthews Correlation Coefficient (MCC):**

```
MCC = (TP×TN - FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

### 5. SHAP Values (Corrected)

**Shapley Value (Game Theory):**

```
φ_i(v) = Σ (|S|!(n-|S|-1)!/n!) * [v(S ∪ {i}) - v(S)]
       S⊆N\{i}
```

dimana:

- `N` = set semua features
- `S` = subset features tidak termasuk feature i
- `v(S)` = nilai koalisi (expected prediction untuk subset S)
- `n` = total jumlah features

**TreeSHAP (untuk tree-based models):**

```
φ_i = Σ Σ (|S|!(M-|S|-1)!/M!) * [f_x(S ∪ {i}) - f_x(S)]
     T∈Trees S⊆Features\{i}
```

### 6. Ensemble Methods

**Voting Classifier (Hard Voting):**

```
ŷ = mode(h₁(x), h₂(x), ..., h_m(x))
```

**Voting Classifier (Soft Voting):**

```
ŷ = argmax_c Σ w_i * p_i,c(x)
```

dimana:

- `w_i` = weight untuk model i
- `p_i,c(x)` = probabilitas prediksi kelas c dari model i

**Weighted Average Ensemble:**

```
ŷ = Σ w_i * h_i(x) / Σ w_i
```

### 7. Hyperparameter Optimization

**Grid Search:**

```
θ* = argmin_θ∈Θ CV_error(θ)
dimana Θ = θ₁ × θ₂ × ... × θ_d (Cartesian product)
```

**Random Search:**

```
θ* = argmin_θ∈random_sample(Θ,n) CV_error(θ)
```

**Bayesian Optimization (Optuna):**

```
θ*_{t+1} = argmax_θ α(θ|D₁:t)
dimana α adalah acquisition function (EI, UCB, etc.)
```

### 8. Cross-Validation

**K-Fold Cross Validation:**

```
CV_error = (1/k) * Σ L(h^(-i)(x_i), y_i)
```

dimana `h^(-i)` adalah model yang dilatih tanpa fold ke-i

**Stratified K-Fold:**

```
Mempertahankan proporsi kelas yang sama di setiap fold
```

### 9. Data Preprocessing

**Standard Scaler (Z-score normalization):**

```
x_scaled = (x - μ) / σ
dimana μ = mean, σ = standard deviation
```

**Min-Max Scaler:**

```
x_scaled = (x - x_min) / (x_max - x_min)
```

**Robust Scaler:**

```
x_scaled = (x - median) / IQR
dimana IQR = Q75 - Q25
```

### 10. Feature Engineering

**Rolling Average:**

```
MA_t = (1/n) * Σ x_{t-i+1} untuk i = 1 sampai n
```

**Exponential Moving Average:**

```
EMA_t = α * x_t + (1-α) * EMA_{t-1}
```

**Feature Interaction:**

```
x_new = x_i * x_j  [multiplicative]
x_new = x_i + x_j  [additive]
```

## 🚀 Memulai

### 1. Persiapan Lingkungan

```bash
# Clone repository
git clone <repository-url>
cd pump_Predictor-main

# Buat virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Skrip

```bash
# Buat skrip executable
chmod +x scripts/*.sh

# Jalankan setup
./scripts/setup.sh
```

### 3. Persiapan Data

```bash
# Gunakan data yang sudah ada
ls data/pump_data.csv

# Atau generate data baru untuk testing
python generate_dataset.py
```

### 4. Latih Model

```bash
# Training dasar
./scripts/train_models.sh

# Dengan hyperparameter tuning
./scripts/train_models.sh --tune

# Mode cepat (training lebih cepat)
./scripts/train_models.sh --quick

# Model kustom
./scripts/train_models.sh --models rf,xgb
```

### 5. Jalankan Layanan

```bash
# Mulai API dan dashboard
./scripts/start_services.sh

# Hentikan layanan
./scripts/stop_services.sh
```

## 🔧 Quick Start Guide

### 1. Instalasi Cepat

```bash
# Clone dan setup
git clone <repository-url>
cd pump_Predictor-main

# Auto setup dengan script
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### 2. Generate Dataset (Opsional)

```bash
# Generate data baru untuk testing
python generate_dataset.py
```

### 3. Training Model

```bash
# Training basic
./scripts/train_models.sh

# Training dengan tuning
./scripts/train_models.sh --tune
```

### 4. Jalankan Aplikasi

```bash
# Start semua services
./scripts/start_services.sh

# Akses:
# API: http://localhost:8000
# Dashboard: http://localhost:8501
```

## 📋 Format Data

File CSV input harus memiliki kolom berikut:

| Kolom                 | Tipe   | Deskripsi              | Rentang             |
| --------------------- | ------ | ---------------------- | ------------------- |
| `timestamp`           | string | Waktu pengukuran       | YYYY-MM-DD HH:MM:SS |
| `pump_id`             | string | ID pompa               | PUMP_001 format     |
| `temperature`         | float  | Suhu operasi (°C)      | 30-100              |
| `pressure`            | float  | Tekanan operasi (bar)  | 50-250              |
| `vibration`           | float  | Frekuensi getaran (Hz) | 0-8                 |
| `flow_rate`           | float  | Laju aliran (L/min)    | 150-400             |
| `motor_current`       | float  | Arus motor (A)         | 5-25                |
| `bearing_temperature` | float  | Suhu bearing (°C)      | 25-90               |
| `oil_level`           | float  | Level oli (%)          | 20-100              |
| `power_consumption`   | float  | Konsumsi daya (kW)     | 5-50                |
| `efficiency`          | float  | Efisiensi pompa (%)    | 0.5-0.95            |
| `operating_hours`     | float  | Jam operasi            | 0-8760              |
| `load_factor`         | float  | Faktor beban           | 0.3-1.0             |
| `ambient_temperature` | float  | Suhu lingkungan (°C)   | 15-45               |
| `humidity`            | float  | Kelembaban (%)         | 30-90               |
| `needs_maintenance`   | int    | Target (0/1)           | 0,1                 |

**Contoh data:**

```csv
timestamp,pump_id,temperature,pressure,vibration,flow_rate,motor_current,bearing_temperature,oil_level,power_consumption,efficiency,operating_hours,load_factor,ambient_temperature,humidity,needs_maintenance
2025-01-01 00:00:00,PUMP_001,65.2,145.8,2.1,245.3,12.5,58.1,85.2,18.7,0.82,1245.5,0.75,22.1,45.3,0
2025-01-01 00:30:00,PUMP_001,68.1,148.2,2.3,243.1,12.8,59.5,84.8,19.2,0.81,1246.0,0.77,22.3,45.8,0
2025-01-01 01:00:00,PUMP_002,81.5,125.3,3.5,215.8,15.2,72.1,78.3,22.1,0.72,2156.3,0.68,23.5,47.2,1
```

## 🎯 Fitur-Fitur

### 🤖 Model Machine Learning

- **Random Forest**: Ensemble berbasis voting mayoritas
- **XGBoost**: Gradient boosting dengan regularisasi
- **LightGBM**: Fast gradient boosting untuk dataset besar
- **Ensemble Model**: Kombinasi optimal dari semua model

### 📊 Pemrosesan Data

- **Preprocessing**: StandardScaler, penanganan missing value
- **Feature Engineering**: Rolling averages, interaction features
- **Validasi Data**: Type checking, validasi rentang nilai
- **Data Sintetis**: Generator untuk testing dan development

### 🔍 Explainable AI

- **SHAP Values**: Penjelasan prediksi individual
- **Feature Importance**: Ranking fitur secara global
- **Partial Dependence**: Analisis efek fitur
- **LIME**: Interpretasi lokal untuk instance tertentu

### 📈 Visualisasi

- **Metrik Kinerja**: Confusion matrix, kurva ROC/AUC
- **Analisis Fitur**: Plot importance, distribusi data
- **Perbandingan Model**: Radar chart, bar plot
- **Dashboard Interaktif**: Visualisasi berbasis Plotly

### 🔧 Hyperparameter Tuning

- **Grid Search**: Pencarian parameter exhaustive
- **Random Search**: Sampling parameter acak
- **Optuna**: Optimisasi Bayesian untuk efisiensi
- **Auto Tuning**: Pemilihan parameter otomatis

### 🌐 API & Interface Web

- **FastAPI**: RESTful API endpoints dengan dokumentasi otomatis
- **Streamlit**: Dashboard web interaktif
- **Batch Processing**: Prediksi multiple sekaligus
- **Real-time Monitoring**: Monitoring prediksi live

### 📋 Pelaporan

- **Laporan HTML**: Analisis komprehensif dengan visualisasi
- **Export JSON**: Metadata model dan hasil
- **Generasi PDF**: Laporan profesional untuk stakeholder
- **Penjadwalan Otomatis**: Laporan berkala

## 🖥️ Contoh Penggunaan

### Python API

```python
from pump_predictor.main import PumpPredictorPipeline

# Inisialisasi pipeline
pipeline = PumpPredictorPipeline()

# Jalankan pipeline lengkap
results = pipeline.run_complete_pipeline(
    data_path="data/pump_data.csv",
    tune_hyperparams=True,
    model_types=['random_forest', 'xgboost'],
    create_ensemble=True
)

# Lakukan prediksi
from pump_predictor.api.main import ModelManager
model_manager = ModelManager()
model_manager.load_model()

prediksi = model_manager.predict([145.8, 65.2, 2.1, 245.3])
print(f"Perlu maintenance: {prediksi['needs_maintenance']}")
```

### REST API

```bash
# Health check
curl http://localhost:8000/health

# Prediksi tunggal (dengan data yang lebih lengkap)
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 65.2,
    "pressure": 145.8,
    "vibration": 2.1,
    "flow_rate": 245.3,
    "motor_current": 12.5,
    "bearing_temperature": 58.1,
    "oil_level": 85.2,
    "power_consumption": 18.7,
    "efficiency": 0.82,
    "operating_hours": 1245.5,
    "load_factor": 0.75,
    "ambient_temperature": 22.1,
    "humidity": 45.3
  }'

# Prediksi batch
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"data": [{"pressure": 145.8, "temperature": 65.2, "vibration": 2.1, "flow_rate": 245.3}]}'
```

### Dashboard Streamlit

```bash
# Jalankan dashboard
streamlit run pump_predictor/streamlit_app/app.py

# Akses: http://localhost:8501
```

## 📊 Kinerja Model

Metrik kinerja tipikal:

| Model         | Akurasi | Presisi | Recall | F1-Score |
| ------------- | ------- | ------- | ------ | -------- |
| Random Forest | 0.92    | 0.89    | 0.94   | 0.91     |
| XGBoost       | 0.94    | 0.91    | 0.96   | 0.93     |
| LightGBM      | 0.93    | 0.90    | 0.95   | 0.92     |
| Ensemble      | 0.95    | 0.93    | 0.97   | 0.95     |

## 🔧 Konfigurasi

Edit [`pump_predictor/config.py`](pump_predictor/config.py) untuk mengatur:

- Parameter model
- Kolom fitur
- Pengaturan API
- Konfigurasi logging

## 📝 Logging

Sistem logging komprehensif:

- File log: `logs/pump_predictor_YYYYMMDD.log`
- Output console dengan warna
- Structured JSON logging untuk production
- Rotation dan retention policies

## 🐳 Deployment Docker

```bash
# Build dan jalankan dengan Docker Compose
cd deployment/docker
docker-compose up -d

# Scale layanan
docker-compose up -d --scale api=3

# Lihat logs
docker-compose logs -f
```

## ☸️ Kubernetes

```bash
# Deploy ke Kubernetes cluster
kubectl apply -f deployment/kubernetes/

# Cek status deployment
kubectl get pods

# Akses layanan
kubectl port-forward service/pump-predictor-api-service 8000:80
```

## 🧪 Testing

```bash
# Jalankan semua unit test
pytest tests/ -v

# Laporan coverage
pytest tests/ --cov=pump_predictor --cov-report=html

# Test spesifik
pytest tests/test_models.py::TestRandomForestModel -v

# Test dengan output detail
pytest tests/ -v -s
```

## 📚 Fitur Lanjutan

### Monitoring Model

- **Deteksi Data Drift**: Uji statistik untuk perubahan distribusi data
- **Monitoring Kinerja**: Metrik real-time dan alerting
- **Peringatan**: Notifikasi berbasis threshold
- **A/B Testing**: Perbandingan kinerja model

### Integrasi MLOps

- **Versioning Model**: Versioning otomatis dengan metadata
- **Experiment Tracking**: Logging parameter dan hasil
- **CI/CD Pipeline**: Deployment otomatis
- **Model Registry**: Penyimpanan terpusat untuk model

### Analisis Mendalam

- **Analisis Korelasi**: Heatmap korelasi antar fitur
- **Outlier Detection**: Identifikasi data anomali
- **Time Series Analysis**: Analisis trend temporal
- **Feature Selection**: Pemilihan fitur optimal

## 🛠️ Troubleshooting

### Masalah Umum

**1. Model tidak dapat dimuat**

```bash
# Pastikan model sudah dilatih
ls -la models/

# Re-train jika perlu
./scripts/train_models.sh --quick
```

**2. API tidak dapat diakses**

```bash
# Cek apakah layanan berjalan
curl http://localhost:8000/health

# Restart layanan
./scripts/stop_services.sh
./scripts/start_services.sh
```

**3. Error import module**

```bash
# Pastikan PYTHONPATH sudah diset
export PYTHONPATH=/path/to/pump_Predictor-main

# Atau install dalam development mode
pip install -e .
```

## 📊 Struktur Data Sensor

### Sensor Utama

- **Pressure (bar)**: 100-200 (normal), >200 (tinggi), <100 (rendah)
- **Temperature (°C)**: 40-80 (normal), >80 (overheating), <40 (underload)
- **Vibration (Hz)**: 1-3 (normal), >4 (masalah bearing), <1 (underload)
- **Flow Rate (L/min)**: 200-300 (normal), <200 (blockage), >300 (overflow)

### Interpretasi Hasil

- **Maintenance Needed = True**: Pompa memerlukan inspeksi/maintenance
- **Maintenance Needed = False**: Pompa dalam kondisi operasi normal
- **Confidence Score**: Tingkat kepercayaan prediksi (0-1)

## 🤝 Kontribusi

1. Fork repository ini
2. Buat feature branch (`git checkout -b feature/fitur-baru`)
3. Commit perubahan (`git commit -am 'Tambah fitur baru'`)
4. Push ke branch (`git push origin feature/fitur-baru`)
5. Buat Pull Request

### Guidelines Kontribusi

- Tulis unit test untuk fitur baru
- Update dokumentasi jika diperlukan
- Follow PEP 8 style guide
- Tambahkan docstring untuk fungsi/kelas baru

## 📄 Lisensi

Proyek ini dilisensikan di bawah MIT License - lihat file [LICENSE](LICENSE) untuk detail.

## 📞 Dukungan

Untuk dukungan dan pertanyaan:

- Buat issue di repository ini
- Cek dokumentasi di folder [docs/](docs/)
- Review existing issues dan discussions

## 🔗 Link Penting

- **Dokumentasi Lengkap**: [docs/](docs/)
- **API Documentation**: http://localhost:8000/docs
- **Dashboard Web**: http://localhost:8501
- **Laporan Model**: [reports/](reports/)
- **Dataset Sample**: [data/](data/)

## 🚀 Update Log & Improvements

### Versi 2025.06.28

**✨ Fitur Baru:**

- ✅ Generator dataset sintetis (`generate_dataset.py`) dengan 16 fitur sensor
- ✅ Ensemble model dengan weighted voting
- ✅ SHAP explainability terintegrasi
- ✅ Dashboard Streamlit yang diperbaharui
- ✅ API endpoints yang lebih komprehensif
- ✅ Automated scripts untuk deployment

**🔧 Perbaikan:**

- ✅ Format data yang diperluas (16 fitur vs 4 fitur sebelumnya)
- ✅ File management & cleanup unused files
- ✅ Improved error handling & logging
- ✅ Docker & Kubernetes deployment ready
- ✅ Comprehensive testing suite

**🗑️ Clean-up:**

- ❌ Removed unused legacy files (8 files)
- ❌ Fixed duplicate model explainer
- ❌ Cleaned up temporary files
- ❌ Standardized file naming conventions

## 📞 Support & Kontribusi

**🤝 Berkontribusi:**

1. Fork repository
2. Buat feature branch (`git checkout -b feature/fitur-baru`)
3. Commit & push changes
4. Submit Pull Request

**📧 Support:**

- 🐛 Issues: Gunakan GitHub Issues
- 📖 Docs: Cek folder `docs/`
- 💬 Diskusi: GitHub Discussions

**📋 Checklist untuk Kontributor:**

- [ ] Write unit tests untuk fitur baru
- [ ] Update dokumentasi jika diperlukan
- [ ] Follow PEP 8 style guide
- [ ] Tambahkan docstring untuk fungsi/kelas baru
- [ ] Test dengan `pytest tests/ -v`

---

**🏆 Teknologi yang Digunakan:**

| Kategori           | Tools                                          |
| ------------------ | ---------------------------------------------- |
| **ML & Data**      | scikit-learn, XGBoost, LightGBM, pandas, numpy |
| **Explainability** | SHAP, LIME, feature importance                 |
| **Web Framework**  | FastAPI, Streamlit                             |
| **Deployment**     | Docker, Kubernetes, uvicorn                    |
| **Testing**        | pytest, coverage                               |
| **Optimization**   | Optuna, GridSearch, RandomSearch               |
| **Visualization**  | matplotlib, seaborn, plotly                    |

---

**📈 Roadmap 2025:**

- 🔮 **Q3 2025**: Real-time streaming predictions
- 🤖 **Q4 2025**: AutoML integration
- 📊 **Q1 2026**: Advanced anomaly detection
- 🌍 **Q2 2026**: Multi-language support

**⭐ Jika proyek ini membantu Anda, berikan star di GitHub!**

---

© 2025 Pump Predictor Project. Licensed under MIT License.

## 📁 Struktur Proyek

```plaintext
pump_Predictor-main/
├── 📊 data/
│   └── pump_data.csv           # Dataset pompa (2000 samples, 16 fitur)
├── 🗂️ dashboard/
│   └── streamlit_app.py        # Dashboard web external
├── 🚀 deployment/
│   ├── docker/                 # Docker configuration
│   └── kubernetes/             # K8s deployment files
├── 📁 logs/                    # Application logs
├── 💾 models/                  # Trained model files
├── 🧪 tests/                   # Unit tests
│   ├── test_api.py            # API testing
│   ├── test_models.py         # Model testing
│   └── test_preprocessing.py  # Data preprocessing tests
├── 📊 reports/                 # Generated reports & plots
├── 🔧 scripts/                 # Utility scripts
│   ├── setup.sh              # Environment setup
│   ├── train_models.sh       # Model training
│   ├── start_services.sh     # Start API & dashboard
│   └── stop_services.sh      # Stop services
├── 🏗️ pump_predictor/          # Main package
│   ├── api/                   # FastAPI endpoints
│   ├── data/                  # Data processing
│   ├── models/                # ML models
│   ├── streamlit_app/         # Internal dashboard
│   └── utils/                 # Utilities & tools
├── generate_dataset.py         # Generate synthetic data
├── setup.py                   # Package setup
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🗑️ File Management & Clean-up

### File yang Dapat Dihapus Dengan Aman

Berdasarkan analisis dependency, file-file berikut **tidak digunakan** dan dapat dihapus:

```bash
# File debug dan temporary
rm debug_import.py
rm -rf temp/

# File model legacy yang tidak terpakai
rm pump_predictor/model.py
rm pump_predictor/data_handler.py

# File duplikat dan tidak standar
rm pump_predictor/models/model_explainer.py  # Duplikat dari utils/
rm pump_predictor/data/data_validator.py     # Import dikomentar
rm pump_predictor/data/init.py               # Nama file tidak standar
rm pump_predictor/models/init.py             # Nama file tidak standar
rm pump_predictor/utils/init.py              # Nama file tidak standar

# File test duplikat (perhatikan spasi di path)
rm " tests/test_models.py"
```

### File yang HARUS Dipertahankan

- ✅ `generate_dataset.py` - Untuk membuat data testing
- ✅ Semua file di `pump_predictor/models/` kecuali `model_explainer.py`
- ✅ Semua file di `pump_predictor/utils/`
- ✅ Semua file API dan konfigurasi
- ✅ Scripts dan deployment files
