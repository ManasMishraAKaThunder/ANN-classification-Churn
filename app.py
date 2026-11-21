
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# --------------------------------
# LOAD TRAINED MODEL AND ENCODERS
# --------------------------------

@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model("model.h5")

    with open("label_encoder_gender.pkl", "rb") as f:
        label_encoder_gender = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("onehot_encoder_geo.pkl", "rb") as f:
        onehot_encoder_geo = pickle.load(f)

    return model, label_encoder_gender, scaler, onehot_encoder_geo


try:
    model, label_encoder_gender, scaler, onehot_encoder_geo = load_resources()

except Exception as e:
    st.error(f"Error loading model or encoders: {e}")
    st.stop()


# --------------------------------
# STREAMLIT APPLICATION
# --------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction")
st.write("Enter customer details to predict whether the customer is likely to leave the bank.")

st.divider()

# --------------------------------
# CUSTOMER INPUT FORM
# --------------------------------

with st.form("customer_form"):

    st.subheader("Customer Information")

    geography = st.selectbox(
        "Select Geography",
        onehot_encoder_geo.categories_[0]
    )

    gender = st.selectbox(
        "Select Gender",
        label_encoder_gender.classes_
    )

    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=650
        )

        age = st.slider(
            "Enter Age",
            18, 92, 30
        )

        tenure = st.slider(
            "Tenure (Years)",
            0, 10, 5
        )

        balance = st.number_input(
            "Enter Balance",
            min_value=0.0,
            value=0.0
        )

    with col2:
        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=50000.0
        )

        num_products = st.slider(
            "Number of Products",
            1, 4, 1
        )

        has_cr_card = st.selectbox(
            "Has Credit Card?",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        is_active_member = st.selectbox(
            "Is Active Member?",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    submit = st.form_submit_button(
        "Predict Customer Churn",
        use_container_width=True
    )


# --------------------------------
# PREDICTION
# --------------------------------

if submit:

    try:
        input_data = pd.DataFrame({
            "CreditScore": [credit_score],
            "Gender": [label_encoder_gender.transform([gender])[0]],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_products],
            "HasCrCard": [has_cr_card],
            "IsActiveMember": [is_active_member],
            "EstimatedSalary": [estimated_salary]
        })

        # One-hot encode Geography

        geo_encoded = onehot_encoder_geo.transform(
            [[geography]]
        )

        if hasattr(geo_encoded, "toarray"):
            geo_encoded = geo_encoded.toarray()

        geo_encoded_df = pd.DataFrame(
            geo_encoded,
            columns=onehot_encoder_geo.get_feature_names_out(
                ["Geography"]
            )
        )

        # Combine encoded geography with input data

        input_data = pd.concat(
            [input_data, geo_encoded_df],
            axis=1
        )

        # Maintain the exact feature order used during scaler training

        if hasattr(scaler, "feature_names_in_"):
            input_data = input_data[
                list(scaler.feature_names_in_)
            ]

        # Scale input

        input_data_scaled = scaler.transform(input_data)

        # Make prediction

        prediction = model.predict(
            input_data_scaled,
            verbose=0
        )

        prediction_proba = float(prediction[0][0])

        st.divider()
        st.subheader("Prediction Result")

        st.metric(
            "Churn Probability",
            f"{prediction_proba * 100:.2f}%"
        )

        if prediction_proba > 0.5:

            st.error(
                "⚠️ This customer is likely to churn."
            )

            st.write(
                "The customer has a higher probability of leaving the bank."
            )

        else:

            st.success(
                "✅ This customer is not likely to churn."
            )

            st.write(
                "The customer has a higher probability of staying with the bank."
            )

    except Exception as e:
        st.error(f"Prediction Error: {e}")
