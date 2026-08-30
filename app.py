"""
app.py
------
Car Price Prediction Streamlit app with MySQL-backed authentication.

Flow:
    1. Signup / Login
    2. Home -> choose one of the trained models
    3. Prediction -> enter car details and predict price
    4. Logout -> returns to Login

Required files:
    all_models.pkl
    model_metrics.pkl
    scaler.pkl
    label_encoders.pkl
    feature_order.pkl
    auth.py
    database.py
"""

import os
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

from auth import signup, login


st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered",
)

LOG_FILE = "user_activity_log.txt"


# ---------------------------------------------------------------------------
# Backend logging
# ---------------------------------------------------------------------------
def log_event(event: str, **fields):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    details = " | ".join(f"{k}={v}" for k, v in fields.items())
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} | {event} | {details}\n")


# ---------------------------------------------------------------------------
# Load ML artifacts
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
# Session state
# ---------------------------------------------------------------------------
defaults = {
    "logged_in": False,
    "name": "",
    "email": "",
    "page": "login",
    "selected_model": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def go_to(page):
    st.session_state.page = page


def logout():
    log_event(
        "LOGOUT",
        name=st.session_state.name,
        email=st.session_state.email,
    )

    for key, value in defaults.items():
        st.session_state[key] = value

    st.rerun()


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
if st.session_state.logged_in:

    with st.sidebar:
        st.write(f"👤 **{st.session_state.name}**")
        st.caption(st.session_state.email)

        if st.button("Logout"):
            logout()


# ---------------------------------------------------------------------------
# PAGE 1 — LOGIN / SIGNUP
# ---------------------------------------------------------------------------
if not st.session_state.logged_in:

    st.title("🚗 Car Price Predictor")
    st.subheader("Authentication")

    option = st.radio(
        "Choose an option",
        ["Login", "Signup"],
        horizontal=True,
    )

    # ------------------------- LOGIN -------------------------
    if option == "Login":

        st.markdown("### Login")

        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")

        if submitted:

            if not email.strip() or not password:
                st.error("Please enter your email and password.")

            else:
                try:
                    authenticated = login(
                        email.strip(),
                        password,
                    )

                    if authenticated:
                        st.session_state.logged_in = True
                        st.session_state.email = email.strip()

                        # Current auth.login() returns True/False, so use
                        # the email prefix as the display name.
                        st.session_state.name = email.split("@")[0]

                        st.session_state.page = "home"

                        log_event(
                            "LOGIN",
                            name=st.session_state.name,
                            email=st.session_state.email,
                        )

                        st.success("Login successful! 🎉")
                        st.rerun()

                    else:
                        st.error("Invalid email or password.")

                except Exception as e:
                    st.error(f"Login failed: {e}")

    # ------------------------- SIGNUP -------------------------
    else:

        st.markdown("### Create Account")

        with st.form("signup_form"):
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input(
                "Password",
                type="password",
            )
            submitted = st.form_submit_button("Signup")

        if submitted:

            if not username.strip() or not email.strip() or not password:
                st.error("Please fill in all fields.")

            else:
                try:
                    signup(
                        username.strip(),
                        email.strip(),
                        password,
                    )

                    st.success(
                        "Account created successfully! 🎉 "
                        "You can now login."
                    )

                except Exception as e:
                    st.error(f"Signup failed: {e}")


# ---------------------------------------------------------------------------
# PAGE 2 — MODEL SELECTION / HOME
# ---------------------------------------------------------------------------
elif st.session_state.page == "home":

    st.title("🚗 Car Price Predictor")
    st.success(f"Welcome, {st.session_state.name}! 👋")

    st.write(
        "Choose a model below to see its performance, "
        "then use it to predict a car's price."
    )

    for model_name in all_models:

        with st.container(border=True):

            st.subheader(model_name)

            m = model_metrics[model_name]

            c1, c2, c3 = st.columns(3)

            c1.metric("Test R²", f"{m['test_r2']:.3f}")
            c2.metric("Train R²", f"{m['train_r2']:.3f}")
            c3.metric(
                "Overfit Gap",
                f"{m['train_r2'] - m['test_r2']:.3f}",
            )

            plot_path = f"plot_{model_name}.png"

            if os.path.exists(plot_path):
                st.image(
                    plot_path,
                    caption=(
                        f"{model_name} — "
                        "Actual vs Predicted (Train / Test)"
                    ),
                )

            if st.button(
                f"Use {model_name} for Prediction",
                key=f"select_{model_name}",
            ):

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

    # Safety check in case the page is reached without selecting a model.
    if model_name is None or model_name not in all_models:
        st.warning("Please select a model first.")
        go_to("home")
        st.rerun()

    model = all_models[model_name]

    st.title("🚗 Car Price Predictor")
    st.caption(f"Using: **{model_name}**")

    if st.button("🏠 Go to Home", key="top_home"):
        go_to("home")
        st.rerun()

    st.write(
        "Enter the car's details below to estimate its selling price."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            brand = st.selectbox(
                "Brand",
                sorted(encoders["brand"].classes_),
            )

            fuel = st.selectbox(
                "Fuel Type",
                sorted(encoders["fuel"].classes_),
            )

            seller_type = st.selectbox(
                "Seller Type",
                sorted(encoders["seller_type"].classes_),
            )

            transmission = st.selectbox(
                "Transmission",
                sorted(encoders["transmission"].classes_),
            )

            owner = st.selectbox(
                "Owner",
                list(encoders["owner"].classes_),
            )

        with col2:
            car_age = st.number_input(
                "Car Age (years)",
                min_value=0,
                max_value=40,
                value=5,
            )

            km_driven = st.number_input(
                "Kilometers Driven",
                min_value=0,
                value=50000,
                step=1000,
            )

            mileage = st.number_input(
                "Mileage (km/ltr/kg)",
                min_value=0.0,
                value=18.0,
                step=0.1,
            )

            engine = st.number_input(
                "Engine (CC)",
                min_value=0.0,
                value=1200.0,
                step=50.0,
            )

            max_power = st.number_input(
                "Max Power (bhp)",
                min_value=0.0,
                value=80.0,
                step=1.0,
            )

        seats = st.selectbox(
            "Seats",
            [2, 4, 5, 6, 7, 8, 9, 10, 14],
            index=2,
        )

        predict_clicked = st.form_submit_button(
            "Predict Price"
        )

    if predict_clicked:

        raw_input = {
            "km_driven": km_driven,
            "fuel": encoders["fuel"].transform([fuel])[0],
            "seller_type": encoders["seller_type"].transform(
                [seller_type]
            )[0],
            "transmission": encoders["transmission"].transform(
                [transmission]
            )[0],
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

        st.success(
            f"### Estimated Selling Price: ₹{prediction:,.0f}"
        )

        st.caption(
            "This is a model estimate based on historical "
            "CarDekho listings — actual market price can vary."
        )

    st.markdown("---")

    if st.button("🏠 Go to Home", key="bottom_home"):
        go_to("home")
        st.rerun()
