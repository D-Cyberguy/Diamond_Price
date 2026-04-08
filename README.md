# 💎 Diamond Price Predictor

An end-to-end Machine Learning project that predicts diamond prices based on physical properties. Built with a production-grade architecture including a full sklearn pipeline, FastAPI REST API, and Docker containerization.

---

## 📊 Model Performance

| Model | R² Score | RMSE | MAE |
|---|---|---|---|
| **Tuned XGBoost** ✅ | **0.9928** | **0.0856** | **0.0602** |
| Random Forest | 0.9921 | 0.0918 | 0.0634 |
| Gradient Boosting | 0.9885 | 0.1078 | 0.0819 |
| Decision Tree | 0.9848 | 0.1236 | 0.0840 |
| Ridge | 0.9664 | 0.2074 | 0.1137 |
| Linear Regression | 0.9667 | 0.2082 | 0.1137 |
| Lasso | -0.0002 | 0.9193 | 0.7909 |

> **Note:** Price predictions are log-transformed internally (`log1p`) and reversed (`expm1`) before returning the final USD value.

---

## 🗂️ Project Structure

```
Diamond_Price/
├── artifacts/                  # Generated files — gitignored
│   ├── model.pkl               # Trained XGBoost model
│   ├── preprocessor.pkl        # Fitted sklearn preprocessor
│   ├── raw_data.csv
│   ├── train.csv
│   └── test.csv
├── data/
│   ├── Diamonds Prices2022.csv # Raw dataset
│   ├── Data Dictionary.xlsx
│   └── metadata.txt
├── notebooks/
│   ├── Diamond_Price_EDA.ipynb
│   ├── Diamond_Price_FE.ipynb
│   └── Diamond_Price_Model.ipynb
├── src/
│   ├── __init__.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py
│   ├── logger.py
│   ├── exception.py
│   └── utils.py
├── app.py                      # FastAPI application
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── setup.py
```

---

## ⚙️ Features Used

| Feature | Type | Description |
|---|---|---|
| `carat` | Numeric | Diamond weight |
| `cut` | Ordinal | Fair → Good → Very Good → Premium → Ideal |
| `color` | Ordinal | J (worst) → D (best) |
| `clarity` | Ordinal | I1 → IF |
| `depth` | Numeric | Total depth percentage |
| `table` | Numeric | Width of top facet |
| `zirconia_length` | Numeric | x dimension (renamed) |
| `zirconia_width` | Numeric | y dimension (renamed) |
| `zirconia_height` | Numeric | z dimension (renamed) |
| `volume` | Numeric | x × y × z × 0.0061 (engineered) |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/D-Cyberguy/Diamond_Price.git
cd Diamond_Price
```

### 2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
python -m pip install -r requirements.txt
```

### 4. Run the training pipeline
```bash
python -m src.pipeline.train_pipeline
```

### 5. Start the API
```bash
uvicorn app:app --reload
```

### 6. Visit the interactive docs
```
http://127.0.0.1:8000/docs
```

---

## 🐳 Docker

### Build the image
```bash
docker build -t diamond-price-predictor .
```

### Run the container
```bash
docker run -p 8000:8000 diamond-price-predictor
```

---

## 📡 API Endpoints

### `GET /`
Returns API info and links.

### `GET /health`
Returns `{"status": "ok"}` — used for health checks in CI/CD.

### `POST /predict`
Predict the price of a diamond.

**Request body:**
```json
{
  "carat": 0.23,
  "cut": "Ideal",
  "color": "E",
  "clarity": "SI2",
  "depth": 61.5,
  "table": 55.0,
  "x": 3.95,
  "y": 3.98,
  "z": 2.43
}
```

**Response:**
```json
{
  "predicted_price_usd": 342.50,
  "model_version": "1.0.0",
  "status": "success"
}
```

---

## 🔧 Feature Engineering

The following transformations are applied automatically by the pipeline:

1. **Zero imputation** — Zeros in `x`, `y`, `z` are flagged as missing and imputed using carat-grouped median
2. **Renaming** — `x`, `y`, `z` renamed to `zirconia_length`, `zirconia_width`, `zirconia_height`
3. **Volume feature** — `volume = x × y × z × 0.0061` (correlates ~0.978 with carat)
4. **Duplicate removal** — 149 duplicate rows removed from raw data
5. **Ordinal encoding** — `cut`, `color`, `clarity` encoded with natural quality ordering
6. **Log transformation** — `price` log-transformed with `log1p` to reduce right skew
7. **Standard scaling** — All numeric features scaled with `StandardScaler`

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| Data processing | pandas, numpy |
| ML | scikit-learn, XGBoost |
| API | FastAPI, uvicorn |
| Containerization | Docker |
| Logging | Python logging module |
| Versioning | Git + GitHub |

---

## 👤 Author

**D-Cyberguy**  
Built as a full end-to-end ML engineering project.
