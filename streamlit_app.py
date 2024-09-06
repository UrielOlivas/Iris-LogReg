import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

model = joblib.load('logReg.pkl')
full_pipeline = joblib.load('pipeline.joblib')

def predict(features):
    features_df = pd.DataFrame([features], columns=[
        'sepal_length', 'sepal_width', 'petal_length', 'petal_width'
    ])
    
    # Transformar las features usando el pipeline
    transformed_features = full_pipeline.transform(features_df)
    prediction = model.predict(transformed_features)
    return prediction[0]

st.markdown(
    """
    <h1 style='text-align: center;'>Iris Species Prediction</h1>
    """,
    unsafe_allow_html=True
)
st.image('image.png',use_column_width=True)
st.markdown(
    """
    <div style="text-align: center;">
        <p>Francisco Uriel Olivas Márquez 341948</p>
    </div>
    """,
    unsafe_allow_html=True
)
st.write("### Input Data")
col1, col2 = st.columns(2)
# Inputs
sepal_length = col1.number_input("Longitud del Sépalo", min_value=4.0, max_value=8.0, value=5.1)
sepal_width = col1.number_input("Ancho del Sépalo", min_value=1.5, max_value=5.0, value=3.5)
petal_length = col2.number_input("Longitud del Pétalo", min_value=1.0, max_value=7.0, value=1.4)
petal_width = col2.number_input("Ancho del Pétalo", min_value=0.0, max_value=3.0,value=0.1)


features = [
    sepal_length,
    sepal_width,
    petal_length,
    petal_width,
]

st.write("### Output Data")
if st.button("Predict"):
    prediction = predict(features)
    st.write(f"La especie de flor es: {(prediction.upper())}")