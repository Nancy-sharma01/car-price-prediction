# 🚗 Car Price Prediction

An end-to-end machine learning project that predicts the resale price of used cars, built on the CarDekho dataset. Three regression models are trained, tuned, and compared for overfitting — then served through a multi-page Streamlit app with login, model selection, and instant price prediction.

## ✨ Features

- **Data preprocessing** — handles missing values, duplicates, high-cardinality categorical columns (brand extraction), outlier clipping, label encoding, and feature scaling
- **Three tuned regression models** — Random Forest, Gradient Boosting, and XGBoost, each optimized with `RandomizedSearchCV`
- **Overfitting diagnostics** — train vs test R² comparison, actual-vs-predicted plots, and residual plots for every model
- **Multi-page Streamlit app**:
  - 🔐 Login page (name + email)
  - 📊 Model selection page — compare accuracy and graphs for all three models
  - 💰 Prediction page — enter car details, get an instant price estimate
- **Backend activity logging** — login, model selection, and logout events are logged with timestamps

## 🧠 Model Performance

| Model | Train R² | Test R² | Overfit Gap | Test MAE (₹) |
|---|---|---|---|---|
| Random Forest | 0.983 | 0.889 | 0.094 | 68,380 |
| Gradient Boosting | 0.964 | 0.912 | 0.052 | 63,911 |
| **XGBoost** ✅ | 0.963 | 0.914 | **0.049** | **62,853** |

XGBoost generalizes best (highest test R², smallest train/test gap) and is used as the default model.

## 🛠️ Tech Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · Matplotlib · Seaborn · Streamlit · Joblib

## 📂 Project Structure

```
car-price-prediction/
├── train.py              # Full preprocessing + training + tuning pipeline
├── app.py                 # Multi-page Streamlit app
├── cardekho.csv            # Dataset
├── requirements.txt
└── .gitignore
```

Running `train.py` generates the model artifacts (`all_models.pkl`, `scaler.pkl`, `label_encoders.pkl`, `feature_order.pkl`, `model_metrics.pkl`) and diagnostic plots — these are not committed to the repo since the trained models exceed GitHub's file size limit.

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/Nancy-sharma01/car-price-prediction.git
cd car-price-prediction

# 2. Create a virtual environment & install dependencies
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 3. Train the models (takes a few minutes)
python train.py

# 4. Run the app
streamlit run app.py
```

## 📈 Dataset

[CarDekho Used Car Dataset](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) — 8,128 used car listings with specifications and selling price.

## 🔮 Future Scope

- Replace the plain-text activity log with a database
- Add password-based authentication
- Compare against LightGBM / CatBoost
- Add SHAP-based prediction explanations
- Host publicly on Streamlit Community Cloud

## 👤 Author

**Nancy Sharma** — B.Tech CSE, JMIT Radaur