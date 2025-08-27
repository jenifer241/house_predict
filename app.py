import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Load dataset
def load_data():
    df = pd.read_csv(r"C:\Users\jenif\Downloads\house_prices.csv")   # Make sure file is in same folder
    return df

df = load_data()

st.title("✨ Jenifer’s Price Prediction App")
st.divider()
st.image(r"C:\Users\jenif\Downloads\stephan-bechert-yFV39g6AZ5o-unsplash.jpg",use_container_width=True) 


# Define features and target
X = df[["bedrooms", "bathrooms", "sqft_living"]]
y = df["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)


# User Input for Prediction
col1, col2, col3 = st.columns(3)

with col1:
    bedroom = st.number_input("🛏️ Bedrooms", min_value=1, value=3, step=1)

with col2:
    bathroom = st.number_input("🛁 Bathrooms", min_value=1, value=2, step=1)

with col3:
    sqr = st.number_input("📐 Square Feet", min_value=200, value=1500, step=50)

st.divider()

if st.button("Predict Price"):
    input_data = pd.DataFrame([[bedroom, bathroom, sqr]], columns=["bedrooms", "bathrooms", "sqft_living"])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted House Price: ${prediction:,.2f}")
    st.balloons()
    st.snow()

 # --------- CSV Download ---------
    result = pd.DataFrame({
        "Bedrooms": [bedroom],
        "Bathrooms": [bathroom],
        "Square Feet": [sqr],
        "Predicted Price": [prediction]
    })

    csv = result.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download these file(CSV)",
        data=csv,
        file_name="house_price_prediction.csv",
        mime="text/csv",
    )