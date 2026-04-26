# app.py
# This is the web app. Streamlit turns Python into a website automatically.
# Every time a user changes a slider/dropdown, this whole file re-runs top to bottom.

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# PAGE CONFIG 
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# LOAD MODEL 
# @st.cache_resource means: load this only ONCE, not on every user interaction.
# Without this, the model would reload every time you move a slider — very slow.

@st.cache_resource
def load_model():
    with open("model/churn_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/le_contract.pkl", "rb") as f:
        le_contract = pickle.load(f)
    with open("model/le_payment.pkl", "rb") as f:
        le_payment = pickle.load(f)
    with open("model/features.pkl", "rb") as f:
        features = pickle.load(f)
    return model, le_contract, le_payment, features

model, le_contract, le_payment, features = load_model()

# HEADER 
st.title("📊 Customer Churn Prediction")
st.markdown("Enter customer details below to predict whether they are likely to churn.")
st.divider()

# INPUT FORM
# st.columns splits the page into side-by-side sections
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Account Info")

    tenure = st.slider(
        "Tenure (months)",
        min_value=1, max_value=72, value=12,
        help="How many months has this customer been with the company?"
    )

    contract_type = st.selectbox(
        "Contract Type",
        options=["Month-to-month", "One year", "Two year"],
        help="Month-to-month customers churn more often"
    )

    num_products = st.slider(
        "Number of Products",
        min_value=1, max_value=4, value=2
    )

with col2:
    st.subheader("Billing Info")

    monthly_charges = st.slider(
        "Monthly Charges (₹)",
        min_value=20, max_value=120, value=65
    )

    total_charges = st.number_input(
        "Total Charges (₹)",
        min_value=100, max_value=8000, value=1500, step=100
    )

    payment_method = st.selectbox(
        "Payment Method",
        options=["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    )

with col3:
    st.subheader("Services & Demographics")

    has_internet = st.radio(
        "Internet Service",
        options=[1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    has_phone = st.radio(
        "Phone Service",
        options=[1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    is_senior = st.radio(
        "Senior Citizen",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

st.divider()

# PREDICT 
if st.button("Predict Churn", type="primary", use_container_width=True):

    # Encode categorical inputs the same way we did during training
    contract_encoded = le_contract.transform([contract_type])[0]
    payment_encoded  = le_payment.transform([payment_method])[0]

    # Build input row — ORDER must match features list from training
    input_data = pd.DataFrame([{
        "tenure":           tenure,
        "monthly_charges":  monthly_charges,
        "total_charges":    total_charges,
        "num_products":     num_products,
        "has_internet":     has_internet,
        "has_phone":        has_phone,
        "is_senior":        is_senior,
        "contract_encoded": contract_encoded,
        "payment_encoded":  payment_encoded,
    }])[features]  # reorder columns to match training

    # Get prediction and probability
    prediction   = model.predict(input_data)[0]           # 0 or 1
    probability  = model.predict_proba(input_data)[0]     # [prob_no_churn, prob_churn]
    churn_prob   = probability[1] * 100                   # churn probability as %

    # RESULT DISPLAY 
    st.subheader("Prediction Result")
    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        if prediction == 1:
            st.error(" HIGH CHURN RISK")
            st.markdown("This customer is **likely to leave.**")
        else:
            st.success(" LOW CHURN RISK")
            st.markdown("This customer is **likely to stay.**")

    with res_col2:
        st.metric(
            label="Churn Probability",
            value=f"{churn_prob:.1f}%",
            delta=f"{churn_prob - 50:.1f}% vs average" if churn_prob != 50 else None,
            delta_color="inverse"
        )

    with res_col3:
        st.metric(
            label="Retention Probability",
            value=f"{100 - churn_prob:.1f}%"
        )

    # PROBABILITY BAR 
    st.progress(int(churn_prob), text=f"Churn risk: {churn_prob:.1f}%")

    # FEATURE IMPORTANCE
    # Show which factors influenced this prediction most
    st.subheader("Top Factors Influencing This Prediction")

    importances = model.feature_importances_
    feature_df  = pd.DataFrame({
        "Feature":    ["Tenure", "Monthly Charges", "Total Charges", "No. of Products",
                       "Internet", "Phone", "Senior Citizen", "Contract Type", "Payment Method"],
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    st.bar_chart(feature_df.set_index("Feature")["Importance"])

    # RECOMMENDATION
    st.subheader("Business Recommendation")

    if churn_prob >= 70:
        st.warning("""
        **Immediate Action Required:**
        - Offer a loyalty discount or upgrade
        - Assign a dedicated account manager
        - Consider offering a contract upgrade incentive
        """)
    elif churn_prob >= 40:
        st.info("""
        **Monitor This Customer:**
        - Send a satisfaction survey
        - Offer a small retention incentive
        - Check for recent service complaints
        """)
    else:
        st.success("""
        **Customer Appears Satisfied:**
        - Good candidate for upselling additional products
        - Consider for referral program
        """)

# SIDEBAR 
with st.sidebar:
    st.header("About This App")
    st.markdown("""
    This app predicts whether a telecom customer is likely to cancel their subscription (churn).

    **Model:** Random Forest Classifier

    **Features used:** 9

    **Built with:** Python, Scikit-learn, Streamlit

    **Deployed on:** Streamlit Community Cloud
    """)

    st.divider()
    st.markdown("Built by **Rashi Sharma**")
    st.markdown("[GitHub](https://github.com/rashiii201)")