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
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.big-title {
    text-align: center;
    font-size: 55px;
    font-weight: bold;
    background: linear-gradient(90deg,#00DBDE,#FC00FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #AAAAAA;
    font-size: 18px;
}

.metric-card {
    background: #1E293B;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}

.result-card {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.markdown(
    '<p class="big-title">📊 Customer Churn Prediction</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Predict whether a customer is likely to leave the company using Machine Learning</p>',
    unsafe_allow_html=True
)

st.write("")

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
with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
        width=120
    )

    st.markdown("## 📌 Project Information")

    st.success("🤖 Random Forest Classifier")

    st.markdown("""
### Features

📅 Tenure

💰 Monthly Charges

💳 Total Charges

---

### Developer

👨‍💻 Mony Rachana

🎓 Year 3 Data Processing
""")

# =========================
# DASHBOARD CARDS
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "Random Forest")

with col2:
    st.metric("Features", "3")

with col3:
    st.metric("Goal", "Predict Churn")

st.divider()

# =========================
# INPUTS
# =========================
st.subheader("📝 Customer Information")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider(
        "📅 Tenure (Months)",
        1,
        72,
        12
    )

with col2:
    monthly_charges = st.number_input(
        "💰 Monthly Charges ($)",
        0.0,
        200.0,
        50.0
    )

total_charges = st.number_input(
    "💳 Total Charges ($)",
    0.0,
    10000.0,
    500.0
)

st.write("")

# =========================
# PREDICT BUTTON
# =========================
if st.button("🚀 Predict Churn", use_container_width=True):

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

    st.subheader("📊 Prediction Result")

    if prediction[0] == 1:

        st.error(
            f"⚠️ Customer Likely To Churn\n\n"
            f"Probability: {churn_probability:.2f}%"
        )

        st.progress(int(churn_probability))

    else:

        st.success(
            f"✅ Customer Likely To Stay\n\n"
            f"Probability: {stay_probability:.2f}%"
        )

        st.progress(int(stay_probability))

    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Stay Probability",
            f"{stay_probability:.2f}%"
        )

    with col2:
        st.metric(
            "Churn Probability",
            f"{churn_probability:.2f}%"
        )

    st.write("")

    st.subheader("📋 Customer Details")

    result_df = pd.DataFrame({
        "Feature": [
            "Tenure",
            "Monthly Charges",
            "Total Charges"
        ],
        "Value": [
            tenure,
            monthly_charges,
            total_charges
        ]
    })

    st.dataframe(
        result_df,
        use_container_width=True
    )

# =========================
# FOOTER
# =========================
st.divider()

st.markdown(
"""
<center>
Built with ❤️ using Streamlit, Scikit-Learn & Python
</center>
""",
unsafe_allow_html=True
)