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
    layout="wide",
)

LOG_FILE = "user_activity_log.txt"


# ---------------------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

            html, body, [class*="css"] {
                font-family: 'Inter', sans-serif;
            }

            #MainMenu, footer, header {visibility: hidden;}

            :root {
                --accent: #4f46e5;
                --accent-light: #eef2ff;
                --accent-dark: #3730a3;
                --success: #16a34a;
            }

            .block-container {
                padding-top: 2rem;
                max-width: 1100px;
            }

            /* Hero header */
            .app-hero {
                background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
                border-radius: 18px;
                padding: 2rem 2.2rem;
                margin-bottom: 1.6rem;
                color: white;
                box-shadow: 0 10px 30px -10px rgba(79, 70, 229, 0.45);
            }
            .app-hero h1 {
                margin: 0;
                font-size: 1.9rem;
                font-weight: 800;
                color: white;
            }
            .app-hero p {
                margin: 0.35rem 0 0 0;
                opacity: 0.9;
                font-size: 0.98rem;
            }

            /* Model cards */
            .model-card {
                border: 1px solid #e5e7eb;
                border-radius: 16px;
                padding: 1.4rem 1.5rem 0.6rem 1.5rem;
                margin-bottom: 1.3rem;
                background: white;
                box-shadow: 0 1px 3px rgba(0,0,0,0.04);
                transition: box-shadow 0.15s ease;
            }
            .model-card:hover {
                box-shadow: 0 8px 20px -6px rgba(0,0,0,0.12);
            }
            .model-card-title {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 0.6rem;
            }
            .model-card-title h3 {
                margin: 0;
                font-size: 1.25rem;
                font-weight: 700;
                color: #111827;
            }
            .best-badge {
                background: var(--success);
                color: white;
                font-size: 0.72rem;
                font-weight: 700;
                padding: 0.25rem 0.65rem;
                border-radius: 999px;
                letter-spacing: 0.03em;
            }

            div[data-testid="stMetric"] {
                background: #f9fafb;
                border-radius: 10px;
                padding: 0.6rem 0.8rem 0.5rem 0.8rem;
                border: 1px solid #f0f0f2;
            }
            div[data-testid="stMetricLabel"] { font-size: 0.78rem; }

            /* Buttons */
            div.stButton > button {
                border-radius: 9px;
                font-weight: 600;
                border: 1px solid var(--accent);
                color: var(--accent);
                background: white;
                padding: 0.5rem 1rem;
            }
            div.stButton > button:hover {
                background: var(--accent);
                color: white;
                border: 1px solid var(--accent);
            }
            div.stFormSubmitButton > button, button[kind="primaryFormSubmit"] {
                background: var(--accent) !important;
                color: white !important;
                border-radius: 9px !important;
                font-weight: 700 !important;
                border: none !important;
                padding: 0.6rem 1.2rem !important;
            }
            div.stFormSubmitButton > button:hover {
                background: var(--accent-dark) !important;
            }

            /* Prediction result card */
            .result-card {
                background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
                border: 1px solid #a7f3d0;
                border-radius: 16px;
                padding: 1.4rem 1.6rem;
                margin-top: 1rem;
            }
            .result-card .label {
                font-size: 0.85rem;
                color: #047857;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                margin-bottom: 0.2rem;
            }
            .result-card .value {
                font-size: 2.1rem;
                font-weight: 800;
                color: #065f46;
            }
            .result-card .note {
                margin-top: 0.5rem;
                font-size: 0.83rem;
                color: #047857;
                opacity: 0.85;
            }

            .section-label {
                font-size: 0.78rem;
                font-weight: 700;
                color: #6b7280;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin: 0.4rem 0 0.4rem 0;
            }

            /* Auth card */
            .auth-card {
                border: 1px solid #e5e7eb;
                border-radius: 16px;
                padding: 1.8rem 1.9rem;
                background: white;
                box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


def hero(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="app-hero">
            <h1>🚗 {title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
        st.markdown(
            f"""
            <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
                <div style="
                    width: 64px; height: 64px; border-radius: 50%;
                    background: linear-gradient(135deg, #4f46e5, #7c3aed);
                    color: white; display: flex; align-items: center;
                    justify-content: center; font-size: 1.6rem; font-weight: 700;
                    margin: 0 auto 0.6rem auto;">
                    {st.session_state.name[:1].upper() if st.session_state.name else "?"}
                </div>
                <div style="font-weight:700; font-size:1.05rem;">{st.session_state.name}</div>
                <div style="color:#6b7280; font-size:0.85rem;">{st.session_state.email}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

        if st.session_state.selected_model:
            st.caption("Active model")
            st.markdown(f"**🧠 {st.session_state.selected_model}**")
            st.markdown("---")

        if st.button("🚪 Logout", use_container_width=True):
            logout()


# ---------------------------------------------------------------------------
# PAGE 1 — LOGIN / SIGNUP
# ---------------------------------------------------------------------------
if not st.session_state.logged_in:

    hero("Car Price Predictor", "Sign in to estimate used-car prices with tuned ML models.")

    left, mid, right = st.columns([1, 1.4, 1])

    with mid:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)

        option = st.radio(
            "Choose an option",
            ["Login", "Signup"],
            horizontal=True,
            label_visibility="collapsed",
        )

        # ------------------------- LOGIN -------------------------
        if option == "Login":

            st.markdown("#### Welcome back 👋")

            with st.form("login_form"):
                email = st.text_input("Email", placeholder="you@example.com")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                submitted = st.form_submit_button("Login", use_container_width=True)

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

            st.markdown("#### Create your account 🚀")

            with st.form("signup_form"):
                username = st.text_input("Username", placeholder="Your name")
                email = st.text_input("Email", placeholder="you@example.com")
                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="••••••••",
                )
                submitted = st.form_submit_button("Create account", use_container_width=True)

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

        st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE 2 — MODEL SELECTION / HOME
# ---------------------------------------------------------------------------
elif st.session_state.page == "home":

    hero(
        "Car Price Predictor",
        f"Welcome back, {st.session_state.name} 👋 — pick a model below to get started.",
    )

    best_model = max(model_metrics, key=lambda k: model_metrics[k]["test_r2"])

    for model_name in all_models:

        m = model_metrics[model_name]
        gap = m["train_r2"] - m["test_r2"]
        is_best = model_name == best_model

        st.markdown('<div class="model-card">', unsafe_allow_html=True)

        badge_html = '<span class="best-badge">⭐ BEST TEST R²</span>' if is_best else ""
        st.markdown(
            f"""
            <div class="model-card-title">
                <h3>🧠 {model_name}</h3>
                {badge_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Test R²", f"{m['test_r2']:.3f}")
        c2.metric("Train R²", f"{m['train_r2']:.3f}")
        c3.metric("Overfit Gap", f"{gap:.3f}", delta=None)

        st.progress(min(max(m["test_r2"], 0.0), 1.0))

        plot_path = f"plot_{model_name}.png"

        if os.path.exists(plot_path):
            with st.expander("📊 Actual vs Predicted plots"):
                st.image(
                    plot_path,
                    caption=(
                        f"{model_name} — "
                        "Actual vs Predicted (Train / Test)"
                    ),
                    use_container_width=True,
                )

        st.write("")

        if st.button(
            f"Use {model_name} for Prediction  →",
            key=f"select_{model_name}",
            use_container_width=True,
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

        st.markdown('</div>', unsafe_allow_html=True)


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

    hero("Car Price Predictor", f"Using the <b>{model_name}</b> model to estimate your car's price.")

    top_l, top_r = st.columns([1, 5])
    with top_l:
        if st.button("🏠 Home", key="top_home"):
            go_to("home")
            st.rerun()

    with st.form("prediction_form"):

        st.markdown('<div class="section-label">Vehicle details</div>', unsafe_allow_html=True)
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

        st.markdown('<div class="section-label">Capacity</div>', unsafe_allow_html=True)
        seats = st.selectbox(
            "Seats",
            [2, 4, 5, 6, 7, 8, 9, 10, 14],
            index=2,
            label_visibility="collapsed",
        )

        st.write("")
        predict_clicked = st.form_submit_button(
            "🔮  Predict Price", use_container_width=True,
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

        st.markdown(
            f"""
            <div class="result-card">
                <div class="label">Estimated Selling Price</div>
                <div class="value">₹{prediction:,.0f}</div>
                <div class="note">
                    Model estimate based on historical CarDekho listings —
                    actual market price can vary.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("---")

    if st.button("🏠 Go to Home", key="bottom_home"):
        go_to("home")
        st.rerun()