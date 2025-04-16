import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, RobustScaler

st.title("Fraud Detection Web App")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

# Load pickled model
@st.cache_resource
def load_model():
    with open('svc_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Original Data")
    st.dataframe(df.head())

    try:
        # Step 1: Normalize time and amount
        sk = StandardScaler()
        rs = RobustScaler()
        df['Time'] = sk.fit_transform(df['Time'].values.reshape(-1, 1))
        df['Amount'] = rs.fit_transform(df['Amount'].values.reshape(-1, 1))

        # Step 2: Balance classes
        df = df.sample(frac=1)
        fraud_df = df[df['Class'] == 1]
        non_fraud_df = df[df['Class'] == 0][:492]
        balanced_df = pd.concat([fraud_df, non_fraud_df]).sample(frac=1, random_state=42)

        # Step 3: Split features and target
        X = balanced_df.iloc[:, :-1].values
        Y = balanced_df.iloc[:, -1].values

        # Step 4: Predict using model
        y_pred = model.predict(X)

        # Add predictions to DataFrame
        balanced_df['Predicted'] = y_pred

        st.subheader("Data with Predictions")
        st.dataframe(balanced_df.head())

        # Download button
        csv = balanced_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Result CSV",
            data=csv,
            file_name='predicted_results.csv',
            mime='text/csv'
        )
    except Exception as e:
        st.error(f"Something went wrong: {e}")
