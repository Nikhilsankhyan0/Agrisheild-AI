import streamlit as st
import pandas as pd
import numpy as np
# import joblib
from xgboost import XGBClassifier


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AgriShield AI",
    page_icon="🌾",
    layout="wide"
)


# =========================================================
# MODEL PATH
# =========================================================

MODEL_PATH = "model_artifacts"


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    # Risk score model
    risk_model = joblib.load(
        f"{MODEL_PATH}/final_Risk_score_model.pkl"
    )

    # Scaler for risk score model
    regression_scaler = joblib.load(
        f"{MODEL_PATH}/regression_scaler.pkl"
    )

    # XGBoost classifier
    classifier = XGBClassifier()

    classifier.load_model(
        f"{MODEL_PATH}/final_xgb_classifier.json"
    )

    # Loan status encoder
    label_encoder = joblib.load(
        f"{MODEL_PATH}/classification_label_encoder.pkl"
    )

    return (
        risk_model,
        regression_scaler,
        classifier,
        label_encoder
    )


# =========================================================
# LOAD MODELS
# =========================================================

try:

    (
        risk_model,
        regression_scaler,
        classifier,
        label_encoder
    ) = load_models()

except Exception as e:

    st.error(f"Unable to load models: {e}")
    st.stop()


# =========================================================
# TITLE
# =========================================================

st.title("🌾 AgriShield AI")

st.subheader(
    "AI-Based Agricultural Loan Risk Assessment"
)

st.write(
    "Enter farmer and loan information to assess loan eligibility."
)

st.success("✅ Trained models loaded successfully!")


# =========================================================
# FARMER INFORMATION
# =========================================================

st.header("👨‍🌾 Farmer Information")

col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

with col2:

    family_size = st.number_input(
        "Family Size",
        min_value=1,
        max_value=20,
        value=4
    )

with col3:

    dependents = st.number_input(
        "Dependents",
        min_value=0,
        max_value=15,
        value=2
    )


# =========================================================
# FARM INFORMATION
# =========================================================

st.header("🌱 Farm Information")

col1, col2, col3 = st.columns(3)

with col1:

    farm_size = st.number_input(
        "Farm Size (Acres)",
        min_value=0.0,
        value=5.0
    )

with col2:

    farming_experience = st.number_input(
        "Farming Experience (Years)",
        min_value=0,
        max_value=80,
        value=10
    )

with col3:

    productivity = st.number_input(
        "Farm Productivity Index",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )


# =========================================================
# FINANCIAL INFORMATION
# =========================================================

st.header("💰 Financial Information")

col1, col2, col3 = st.columns(3)

with col1:

    income = st.number_input(
        "Annual Farm Income",
        min_value=0.0,
        value=200000.0
    )

with col2:

    expenses = st.number_input(
        "Annual Farm Expenses",
        min_value=0.0,
        value=100000.0
    )

with col3:

    assets = st.number_input(
        "Asset Value",
        min_value=0.0,
        value=300000.0
    )


# =========================================================
# LOAN INFORMATION
# =========================================================

st.header("🏦 Loan Information")

col1, col2, col3 = st.columns(3)

with col1:

    loan_amount = st.number_input(
        "Loan Amount Requested",
        min_value=0.0,
        value=100000.0
    )

with col2:

    existing_loan = st.number_input(
        "Existing Loan Amount",
        min_value=0.0,
        value=0.0
    )

with col3:

    existing_loans = st.number_input(
        "Existing Loan Count",
        min_value=0,
        max_value=20,
        value=0
    )


# =========================================================
# CREDIT INFORMATION
# =========================================================

st.header("📊 Credit Information")

col1, col2, col3 = st.columns(3)

with col1:

    credit_score = st.number_input(
        "Credit Score",
        min_value=0,
        max_value=900,
        value=650
    )

with col2:

    repayment_score = st.number_input(
        "Repayment History Score",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )

with col3:

    debt_income = st.number_input(
        "Debt to Income Ratio",
        min_value=0.0,
        max_value=10.0,
        value=0.30
    )


# =========================================================
# ASSESS BUTTON
# =========================================================

st.divider()

assess = st.button(
    "🔍 Assess Loan Risk",
    type="primary",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if assess:

    st.info("🔄 Analyzing farmer information...")

    try:

        # =================================================
        # GET FEATURES EXPECTED BY XGBOOST
        # =================================================

        classifier_features = (
            classifier.get_booster().feature_names
        )

        if classifier_features is None:

            st.error(
                "Could not determine classifier features."
            )

            st.stop()


        # =================================================
        # CREATE CLASSIFIER INPUT
        # =================================================

        classifier_input = pd.DataFrame(
            0.0,
            index=[0],
            columns=classifier_features
        )


        # =================================================
        # USER INPUT VALUES
        # =================================================

        values = {

            "Age": age,

            "Family_Size": family_size,

            "Dependents": dependents,

            "Farming_Experience_Years":
                farming_experience,

            "Farm_Size_Acres":
                farm_size,

            "Farm_Productivity_Index":
                productivity,

            "Annual_Farm_Income":
                income,

            "Annual_Farm_Expenses":
                expenses,

            "Asset_Value":
                assets,

            "Loan_Amount_Requested":
                loan_amount,

            "Existing_Loan_Amount":
                existing_loan,

            "Existing_Loan_Count":
                existing_loans,

            "Credit_Score":
                credit_score,

            "Repayment_History_Score":
                repayment_score,

            "Debt_to_Income_Ratio":
                debt_income
        }


        # =================================================
        # PUT VALUES INTO MODEL INPUT
        # =================================================

        for column, value in values.items():

            if column in classifier_input.columns:

                classifier_input.loc[
                    0, column
                ] = value


        # =================================================
        # CONVERT TO NUMPY
        # =================================================

        model_input = classifier_input.to_numpy(
            dtype=np.float32
        )


        # =================================================
        # LOAN CLASSIFICATION
        # =================================================

        predicted_encoded = classifier.predict(
            model_input
        )[0]


        # Convert number to loan status

        predicted_status = (
            label_encoder.inverse_transform(
                [int(predicted_encoded)]
            )[0]
        )


        # =================================================
        # PROBABILITIES
        # =================================================

        probabilities = classifier.predict_proba(
            model_input
        )[0]


        # =================================================
        # RISK SCORE
        # =================================================

        risk_score = None

        try:

            # Features expected by the scaler

            if hasattr(
                regression_scaler,
                "feature_names_in_"
            ):

                regression_features = (
                    regression_scaler.feature_names_in_
                )

                risk_input = pd.DataFrame(
                    0.0,
                    index=[0],
                    columns=regression_features
                )

                for column, value in values.items():

                    if column in risk_input.columns:

                        risk_input.loc[
                            0, column
                        ] = value

                risk_scaled = (
                    regression_scaler.transform(
                        risk_input
                    )
                )

            else:

                risk_scaled = (
                    regression_scaler.transform(
                        model_input
                    )
                )

            risk_score = risk_model.predict(
                risk_scaled
            )[0]

        except Exception as risk_error:

            st.warning(
                f"Risk score could not be calculated: "
                f"{risk_error}"
            )


        # =================================================
        # RESULT
        # =================================================

        st.divider()

        st.header("📋 Loan Assessment Result")


        # =================================================
        # LOAN DECISION
        # =================================================

        if predicted_status == "Approved":

            st.success(
                f"✅ Loan Status: {predicted_status}"
            )

        elif predicted_status == "Approved with Conditions":

            st.warning(
                f"⚠️ Loan Status: {predicted_status}"
            )

        else:

            st.error(
                f"❌ Loan Status: {predicted_status}"
            )


        # =================================================
        # RISK SCORE
        # =================================================

        if risk_score is not None:

            st.metric(
                "Predicted Risk Score",
                f"{risk_score:.2f}"
            )


        # =================================================
        # PREDICTION CONFIDENCE
        # =================================================

        st.subheader("📊 Prediction Confidence")

        probability_df = pd.DataFrame({

            "Loan Status":
                label_encoder.classes_,

            "Probability (%)":
                probabilities * 100

        })

        probability_df["Probability (%)"] = (
            probability_df["Probability (%)"]
            .round(2)
        )

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True
        )


        # =================================================
        # IMPORTANT FACTORS
        # =================================================

        st.subheader("🔎 Important Financial Factors")

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                f"**Credit Score:** {credit_score}"
            )

            st.write(
                f"**Annual Income:** ₹{income:,.0f}"
            )

            st.write(
                f"**Annual Expenses:** ₹{expenses:,.0f}"
            )

            st.write(
                f"**Existing Loan:** ₹{existing_loan:,.0f}"
            )

        with col2:

            st.write(
                f"**Loan Requested:** ₹{loan_amount:,.0f}"
            )

            st.write(
                f"**Debt-to-Income Ratio:** "
                f"{debt_income:.2f}"
            )

            st.write(
                f"**Repayment History:** "
                f"{repayment_score:.1f}/100"
            )

            st.write(
                f"**Asset Value:** ₹{assets:,.0f}"
            )


        # =================================================
        # WHY?
        # =================================================

        st.subheader("💡 Why this decision?")


        # Simple explanation based on supplied
        # financial indicators.

        reasons = []

        if credit_score >= 700:

            reasons.append(
                "Strong credit score supports loan eligibility."
            )

        elif credit_score < 600:

            reasons.append(
                "Low credit score increases lending risk."
            )

        else:

            reasons.append(
                "Credit score is in a moderate range."
            )


        if debt_income <= 0.30:

            reasons.append(
                "Debt-to-income ratio is relatively low."
            )

        elif debt_income > 0.50:

            reasons.append(
                "High debt-to-income ratio increases financial risk."
            )

        else:

            reasons.append(
                "Debt-to-income ratio is at a moderate level."
            )


        if repayment_score >= 75:

            reasons.append(
                "Good repayment history supports the application."
            )

        elif repayment_score < 50:

            reasons.append(
                "Weak repayment history increases loan risk."
            )

        else:

            reasons.append(
                "Repayment history is at a moderate level."
            )


        if income > expenses:

            reasons.append(
                "Farm income is higher than annual expenses."
            )

        else:

            reasons.append(
                "Annual expenses are high compared with income."
            )


        if assets >= loan_amount:

            reasons.append(
                "Assets provide reasonable strength compared with the requested loan."
            )

        else:

            reasons.append(
                "Requested loan is high compared with the reported assets."
            )


        for reason in reasons:

            st.write("•", reason)


        # =================================================
        # DISCLAIMER
        # =================================================

        st.caption(
            "This assessment is generated by trained "
            "machine learning models and should support, "
            "not replace, financial decision-making."
        )


    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )

        st.exception(e)