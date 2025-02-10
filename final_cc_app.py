import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Load the trained model and label encoders
model = joblib.load("churn_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
scaler = joblib.load("scaler.pkl")


# Function to preprocess user input
def preprocess_input(input_data):
    # Convert categorical data using label encoders
    for col in label_encoders:
        if col in input_data:
            input_data[col] = label_encoders[col].transform([input_data[col]])[0]

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Standardize the features
    input_df = scaler.transform(input_df)

    return input_df


# Streamlit app
def main():
    st.title("Customer Churn Prediction")

    # Display the image
    st.image("image.png", caption="Customer Churn Overview", use_container_width=True)

    # User input form
    st.sidebar.header("Customer Details")
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.sidebar.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
    tenure = st.sidebar.number_input(
        "Tenure (months)", min_value=1, max_value=72, value=1
    )
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox(
        "Multiple Lines", ["Yes", "No", "No phone service"]
    )
    internet_service = st.sidebar.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
    )
    online_security = st.sidebar.selectbox(
        "Online Security", ["Yes", "No", "No internet service"]
    )
    online_backup = st.sidebar.selectbox(
        "Online Backup", ["Yes", "No", "No internet service"]
    )
    device_protection = st.sidebar.selectbox(
        "Device Protection", ["Yes", "No", "No internet service"]
    )
    tech_support = st.sidebar.selectbox(
        "Tech Support", ["Yes", "No", "No internet service"]
    )
    streaming_tv = st.sidebar.selectbox(
        "Streaming TV", ["Yes", "No", "No internet service"]
    )
    streaming_movies = st.sidebar.selectbox(
        "Streaming Movies", ["Yes", "No", "No internet service"]
    )
    contract = st.sidebar.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"]
    )
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.sidebar.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    monthly_charges = st.sidebar.number_input(
        "Monthly Charges", min_value=18.25, max_value=118.75, value=50.0
    )
    total_charges = st.sidebar.number_input(
        "Total Charges", min_value=18.8, max_value=8684.8, value=1000.0
    )

    # Map user input to model input
    input_data = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    # Preprocess input data
    input_df = preprocess_input(input_data)

    # Predict churn
    if st.sidebar.button("Predict"):
        prediction = model.predict(input_df)
        churn_rate = model.predict_proba(input_df)[0][1]
        retention_rate = 1 - churn_rate

        st.write(f"Churn Prediction: {'Yes' if prediction[0] == 1 else 'No'}")
        st.write(f"Churn Rate: {churn_rate:.2%}")
        st.write(f"Customer Retention Rate: {retention_rate:.2%}")

        # Display appropriate image based on prediction
        if prediction[0] == 1:  # Churn
            st.image("sad.png", caption="Customer Will Churn", use_container_width=True)
        else:  # No Churn
            st.image(
                "happy.png", caption="Customer Will Stay", use_container_width=True
            )


if __name__ == "__main__":
    main()
