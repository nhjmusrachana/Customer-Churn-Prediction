import streamlit as st
import joblib
import pandas as pd
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# =========================
# TITLE
# =========================
st.title("📊 Customer Churn Prediction")
st.markdown(
    "Predict whether a customer is likely to leave the company based on customer information."
)

# =========================
# LOAD MODEL
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "..",
        "notebook",
        "churn_analysis.ipynb",
        "model",
        "churn_model.pkl"
    )
)

model = joblib.load(MODEL_PATH)

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Project Information")
st.sidebar.info(
    """
    **Model:** Random Forest Classifier
    
    **Features Used:**
    - Tenure
    - Monthly Charges
    - Total Charges
    
    **Goal:**
    Predict customer churn.
    """
)

# =========================
# INPUTS
# =========================
st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider(
        "Tenure (Months)",
        min_value=1,
        max_value=72,
        value=12
    )

with col2:
    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=200.0,
        value=50.0
    )

total_charges = st.number_input(
    "Total Charges ($)",
    min_value=0.0,
    max_value=10000.0,
    value=500.0
)

# =========================
# PREDICT BUTTON
# =========================
if st.button("🔍 Predict Churn"):

    sample = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    prediction = model.predict(sample)
    probability = model.predict_proba(sample)

    churn_probability = probability[0][1] * 100
    stay_probability = probability[0][0] * 100

    st.divider()

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error(
            f"⚠️ Customer Will Churn\n\n"
            f"Churn Probability: {churn_probability:.2f}%"
        )
    else:
        st.success(
            f"✅ Customer Will Stay\n\n"
            f"Stay Probability: {stay_probability:.2f}%"
        )

    st.progress(int(max(churn_probability, stay_probability)))

    st.write("### Prediction Details")

    result_df = pd.DataFrame({
        "Metric": ["Tenure", "Monthly Charges", "Total Charges"],
        "Value": [tenure, monthly_charges, total_charges]
    })

    st.dataframe(result_df, use_container_width=True)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption(
    "Customer Churn Prediction Project | Built with Streamlit, Pandas, Scikit-Learn, and Random Forest"
)   