import streamlit as st
import joblib
import pandas as pd

#set  the tab title 
st.set_page_config("Deployment Project")


#set the page title
st.title("Machine Failure Prediction")

# set Header
st.subheader("By Rutuja Dinkar More")

# Loading Pipelines(Data Cleaning & Data Preprocessing) and Model
pre =joblib.load("pre.joblib")
model = joblib.load("model.joblib")

# Creating input Boxes that take inputs from users(x features)
Type = st.selectbox("Type of Machine Size", ["L", "M", "H"])


# Numeric Inputs with full headings
air_temp = st.number_input("Air Temperature in Kelvin", min_value=250.0, max_value=350.0, value=300.0)
process_temp = st.number_input("Process Temperature in Kelvin", min_value=250.0, max_value=350.0, value=310.0)
rot_speed = st.number_input("Rotational Speed (rpm)", min_value=0, max_value=3000, value=1500)
torque = st.number_input("Torque (Nm)", min_value=0.0, max_value=100.0, value=40.0)
tool_wear = st.number_input("Tool Wear (minutes)", min_value=0, max_value=300, value=50)

# Failure indicators

twf = st.number_input("Tool Wear Failure (TWF)", min_value=0, max_value=1, step=1)
hdf = st.number_input("Heat Dissipation Failure (HDF)", min_value=0, max_value=1, step=1)
pwf = st.number_input("Power Failure (PWF)", min_value=0, max_value=1, step=1)
osf = st.number_input("Overstrain Failure (OSF)", min_value=0, max_value=1, step=1)
rnf = st.number_input("Random Failure (RNF)", min_value=0, max_value=1, step=1)


# Create Temp Difference feature
temp_diff = process_temp - air_temp


# Create dataframe (column names must remain same as training data)
input_data = pd.DataFrame({
    "Type": [Type], # column[]
    "Air temperature [K]": [air_temp],
    "Process temperature [K]": [process_temp],
    "Rotational speed [rpm]": [rot_speed],
    "Torque [Nm]": [torque],
    "Tool wear [min]": [tool_wear],
    "TWF": [twf],
    "HDF": [hdf],
    "PWF": [pwf],
    "OSF": [osf],
    "RNF": [rnf],
    "Temp_diff": [temp_diff]
})


# Prediction button
if st.button("Predict Machine Failure"):
    processed_data = pre.transform(input_data)# when new data comes so wedo pre.transform cleaning, feature scaling
    prediction = model.predict(processed_data)[0]

    if prediction == 1:
        st.error(" Likely a Machine Failure ")
    else:
        st.success(" Machine is Operating Normally") 