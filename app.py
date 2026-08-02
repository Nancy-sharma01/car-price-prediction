"""
app.py
------
Multi-page Streamlit app for Car Selling Price Prediction.

Flow:
    1. Login page       -> captures name + email, shows a greeting
    2. Model selection   -> shows all 3 tuned models with accuracy + graphs
    3. Prediction page   -> input form for the chosen model, shows predicted price
                             + a "Go to Home" button back to model selection

Backend logging:
    Every login, logout, and model selection is appended to
    user_activity_log.txt in the same folder.

Loads artifacts produced by train.py:
    all_models.pkl, model_metrics.pkl, scaler.pkl,
    label_encoders.pkl, feature_order.pkl

Run with:  streamlit run app.py
"""

import os
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

LOG_FILE = "user_activity_log.txt"


# ---------------------------------------------------------------------------
# Backend logging helper
# ---------------------------------------------------------------------------
def log_event(event: str, **fields):
    """Append one line to the .txt activity log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    details = " | ".join(f"{k}={v}" for k, v in fields.items())
    line = f"{timestamp} | {event} | {details}\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)


# ---------------------------------------------------------------------------
# Load saved artifacts (cached so they load once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    all_models = joblib.load("all_models.pkl")
    model_metrics = joblib.load("model_metrics.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("label_encoders.pkl")
    feature_order = joblib.load("feature_order.pkl")
    return all_models, model_metrics, scaler, encoders, feature_order


all_models, model_metrics, scaler, encoders, feature_order = load_artifacts()

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
defaults = {
    "logged_in": False,
    "name": "",
    "email": "",
    "page": "login",
    "selected_model": None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


def go_to(page: str):
    st.session_state.page = page


def logout():
    log_event("LOGOUT", name=st.session_state.name, email=st.session_state.email)
    for key, val in defaults.items():
        st.session_state[key] = val
    st.rerun()


# Sidebar logout control, visible once logged in
if st.session_state.logged_in:
    with st.sidebar:
        st.write(f"👤 **{st.session_state.name}**")
        st.caption(st.session_state.email)
        if st.button("Logout"):
            logout()

# ---------------------------------------------------------------------------
# PAGE 1 — LOGIN
# ---------------------------------------------------------------------------
if st.session_state.page == "login":
    st.title("🚗 Car Price Predictor")
    st.subheader("Login to continue")

    with st.form("login_form"):
        name = st.text_input("Name")
        email = st.text_input("Email ID")
        submitted = st.form_submit_button("Login")

    if submitted:
        if not name.strip() or not email.strip():
            st.error("Please enter both name and email.")
        else:
            st.session_state.logged_in = True
            st.session_state.name = name.strip()
            st.session_state.email = email.strip()
            log_event("LOGIN", name=name.strip(), email=email.strip())
            go_to("home")
            st.rerun()

# ---------------------------------------------------------------------------
# PAGE 2 — MODEL SELECTION (home)
# ---------------------------------------------------------------------------
elif st.session_state.page == "home":
    st.title("🚗 Car Price Predictor")
    st.success(f"Welcome, {st.session_state.name}! 👋")
    st.write("Choose a model below to see its performance, then use it to predict a car's price.")

    for model_name, model in all_models.items():
        with st.container(border=True):
            st.subheader(model_name)

            m = model_metrics[model_name]
            c1, c2, c3 = st.columns(3)
            c1.metric("Test R²", f"{m['test_r2']:.3f}")
            c2.metric("Train R²", f"{m['train_r2']:.3f}")
            c3.metric("Overfit Gap", f"{m['train_r2'] - m['test_r2']:.3f}")

            plot_path = f"plot_{model_name}.png"
            if os.path.exists(plot_path):
                st.image(plot_path, caption=f"{model_name} — Actual vs Predicted (Train / Test)")

            if st.button(f"Use {model_name} for Prediction", key=f"select_{model_name}"):
                st.session_state.selected_model = model_name
                log_event(
                    "MODEL_SELECTED",
                    name=st.session_state.name,
                    email=st.session_state.email,
                    model=model_name,
                )
                go_to("predict")
                st.rerun()

# ---------------------------------------------------------------------------
# PAGE 3 — PREDICTION
# ---------------------------------------------------------------------------
elif st.session_state.page == "predict":
    model_name = st.session_state.selected_model
    model = all_models[model_name]

    st.title("🚗 Car Price Predictor")
    st.caption(f"Using: **{model_name}**")

    if st.button("🏠 Go to Home"):
        go_to("home")
        st.rerun()

    st.write("Enter the car's details below to estimate its selling price.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            brand = st.selectbox("Brand", sorted(encoders["brand"].classes_))
            fuel = st.selectbox("Fuel Type", sorted(encoders["fuel"].classes_))
            seller_type = st.selectbox("Seller Type", sorted(encoders["seller_type"].classes_))
            transmission = st.selectbox("Transmission", sorted(encoders["transmission"].classes_))
            owner = st.selectbox("Owner", list(encoders["owner"].classes_))

        with col2:
            car_age = st.number_input("Car Age (years)", min_value=0, max_value=40, value=5)
            km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000, step=1000)
            mileage = st.number_input("Mileage (km/ltr/kg)", min_value=0.0, value=18.0, step=0.1)
            engine = st.number_input("Engine (CC)", min_value=0.0, value=1200.0, step=50.0)
            max_power = st.number_input("Max Power (bhp)", min_value=0.0, value=80.0, step=1.0)

        seats = st.selectbox("Seats", [2, 4, 5, 6, 7, 8, 9, 10, 14], index=2)

        predict_clicked = st.form_submit_button("Predict Price")

    if predict_clicked:
        raw_input = {
            "km_driven": km_driven,
            "fuel": encoders["fuel"].transform([fuel])[0],
            "seller_type": encoders["seller_type"].transform([seller_type])[0],
            "transmission": encoders["transmission"].transform([transmission])[0],
            "owner": encoders["owner"].transform([owner])[0],
            "mileage(km/ltr/kg)": mileage,
            "engine": engine,
            "max_power": max_power,
            "seats": seats,
            "brand": encoders["brand"].transform([brand])[0],
            "car_age": car_age,
        }

        input_df = pd.DataFrame([raw_input])[feature_order]
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        prediction = max(prediction, 0)

        st.success(f"### Estimated Selling Price: ₹{prediction:,.0f}")
        st.caption(
            "This is a model estimate based on historical CarDekho listings — "
            "actual market price can vary."
        )

    st.markdown("---")
    if st.button("🏠 Go to Home", key="bottom_home"):
        go_to("home")
        st.rerun()