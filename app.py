import streamlit as st
import pandas as pd
import joblib
from transformers import add_interactions
import numpy as np
# Load the trained model
model = joblib.load("CO2 Emissions Prediction.joblib")

# UI Config
st.set_page_config(page_title="CO2 Emissions Predictor", layout="centered")
st.title("CO2 Emissions Predictor")
st.write("Fill out the details to estimate the vehicle's CO2 emissions.")

# --- Dropdown Options ---
transmission_mapping = {
    'A4': 'Automatic - 4-speed',
    'A5': 'Automatic - 5-speed',
    'A6': 'Automatic - 6-speed',
    'A7': 'Automatic - 7-speed',
    'A8': 'Automatic - 8-speed',
    'A9': 'Automatic - 9-speed',
    'AM5': 'Automated Manual - 5-speed',
    'AM6': 'Automated Manual - 6-speed',
    'AM7': 'Automated Manual - 7-speed',
    'AM8': 'Automated Manual - 8-speed',
    'AM9': 'Automated Manual - 9-speed',
    'AS10': 'Automatic with Select Shift - 10-speed',
    'AS4': 'Automatic with Select Shift - 4-speed',
    'AS5': 'Automatic with Select Shift - 5-speed',
    'AS6': 'Automatic with Select Shift - 6-speed',
    'AS7': 'Automatic with Select Shift - 7-speed',
    'AS8': 'Automatic with Select Shift - 8-speed',
    'AS9': 'Automatic with Select Shift - 9-speed',
    'AV': 'Continuously Variable Transmission (CVT)',
    'AV10': 'CVT - 10-speed simulated',
    'AV6': 'CVT - 6-speed simulated',
    'AV7': 'CVT - 7-speed simulated',
    'AV8': 'CVT - 8-speed simulated',
    'M5': 'Manual - 5-speed',
    'M6': 'Manual - 6-speed',
    'M7': 'Manual - 7-speed'
}

vehicle_classes = ['FULL-SIZE', 'MID-SIZE', 'MINICOMPACT', 'MINIVAN', 'PICKUP TRUCK - SMALL',
                   'PICKUP TRUCK - STANDARD', 'SPECIAL PURPOSE VEHICLE', 'STATION WAGON - MID-SIZE',
                   'STATION WAGON - SMALL', 'SUBCOMPACT', 'SUV - SMALL', 'SUV - STANDARD',
                   'TWO-SEATER', 'VAN - CARGO', 'VAN - PASSENGER']

makes = ['FORD', 'TOYOTA', 'HONDA', 'BMW', 'CHEVROLET', 'NISSAN', 'MAZDA', 'VOLKSWAGEN', 'HYUNDAI', 'KIA']  # Sample only
fuel_type_mapping = {
    'Z': 'Regular Gasoline',
    'X': 'Premium Gasoline',
    'E': 'Ethanol (E85)',
    'D': 'Diesel',
    'N': 'Natural Gas'
}

# --- Inputs ---
engine_size = st.number_input("Engine Size (L)", min_value=0.0, max_value=10.0, value=2.0, step=0.1,
                              help="Size of the vehicle's engine in liters (e.g., 2.0 for a 2-liter engine)")
cylinders = st.number_input("Cylinders", min_value=2, max_value=16, value=4,
                            help="Number of engine cylinders; more cylinders often mean more power but higher emissions")
fuel_comb = st.number_input("Fuel Consumption Comb (L/100 km)", min_value=1.0, max_value=30.0, value=8.5, step=0.1,
                            help="Average fuel consumption per 100 km (lower means better efficiency)")
transmission_label = st.selectbox("Transmission", list(transmission_mapping.values()),
                                  help="Select the transmission type (automatic, manual, CVT, etc.)")
transmission = [k for k, v in transmission_mapping.items() if v == transmission_label][0]
vehicle_class = st.selectbox("Vehicle Class", vehicle_classes,
                             help="Category of the vehicle based on size and type (e.g., SUV, Sedan, Van)")
make = st.selectbox("Make", makes)
fuel_label = st.selectbox("Fuel Type", list(fuel_type_mapping.values()),
                          help="Type of fuel the vehicle uses (e.g., Gasoline, Diesel, Ethanol)")
fuel_type = [k for k, v in fuel_type_mapping.items() if v == fuel_label][0]
# --- Predict Button ---
if st.button("Predict CO2 Emissions"):
    # Form DataFrame
    input_df = pd.DataFrame([{
        'Engine Size(L)': engine_size,
        'Cylinders': cylinders,
        'Fuel Consumption Comb (L/100 km)': fuel_comb,
        'Transmission': transmission,
        'Vehicle Class': vehicle_class,
        'Make': make,
        'Fuel Type': fuel_type
    }])

    # Add interaction features
    interactions = add_interactions(input_df)
    full_input = pd.concat([input_df, interactions], axis=1)

    # Predict
    prediction = model.predict(full_input)[0]
    st.success(f"Estimated CO2 Emissions: {prediction:.2f} g/km")
st.markdown("### Model Evaluation Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("R² Score", "0.9943")
col2.metric("MAE", "2.7426 g/km")
col3.metric("RMSE", "4.4085 g/km")

# Show more metrics
st.markdown("""
**Additional Metrics**
- Mean Squared Error (MSE): **19.4351**
- Adjusted R²: **0.9943**
""")



