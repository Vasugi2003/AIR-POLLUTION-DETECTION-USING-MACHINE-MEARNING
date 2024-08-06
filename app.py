import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the model and the label encoder
with open('clf.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)
with open('label_encoder.pkl', 'rb') as encoder_file:
    loaded_label_encoder = pickle.load(encoder_file)

# Define the Streamlit app
st.title('Air Pollution Detection')

# Collect user input
PM25 = st.number_input('PM2.5', min_value=0.0, max_value=1000.0, value=10.0)
PM10 = st.number_input('PM10', min_value=0.0, max_value=1000.0, value=20.0)
NO = st.number_input('NO', min_value=0.0, max_value=1000.0, value=0.1)
NO2 = st.number_input('NO2', min_value=0.0, max_value=1000.0, value=0.2)
NOx = st.number_input('NOx', min_value=0.0, max_value=1000.0, value=0.3)
NH3 = st.number_input('NH3', min_value=0.0, max_value=1000.0, value=0.4)
CO = st.number_input('CO', min_value=0.0, max_value=1000.0, value=0.5)
SO2 = st.number_input('SO2', min_value=0.0, max_value=1000.0, value=0.6)
O3 = st.number_input('O3', min_value=0.0, max_value=1000.0, value=0.7)
Benzene = st.number_input('Benzene', min_value=0.0, max_value=1000.0, value=0.8)
Toluene = st.number_input('Toluene', min_value=0.0, max_value=1000.0, value=0.9)
Xylene = st.number_input('Xylene', min_value=0.0, max_value=1000.0, value=1.0)
AQI = st.number_input('AQI', min_value=0.0, max_value=1000.0, value=50.0)

# Define the feature names
feature_names = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']

# Predict button
if st.button('Predict'):
    # Create a DataFrame for the input data with the correct feature names
    input_data = pd.DataFrame([[PM25, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene, AQI]], columns=feature_names)
    
    # Standardize the input data
    scaler = StandardScaler()
    input_data = scaler.fit_transform(input_data)
    
    # Make prediction
    prediction = loaded_model.predict(input_data)
    prediction_label = loaded_label_encoder.inverse_transform(prediction)[0]
    
    # Display prediction
    st.write(f'Predicted AQI Bucket: {prediction_label}')
