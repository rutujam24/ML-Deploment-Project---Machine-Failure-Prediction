from flask import Flask,render_template,request
import pandas as pd
import joblib

#Create Flask App
flask_app =Flask(__name__)

# Loading the joblib model & preprocessing files
pre = joblib.load("pre.joblib")
model = joblib.load("model.joblib")

#Decorater of flask app used for Home method function
@flask_app.route("/")
def Home():
    return render_template("machine_failure.html")  # Till this was home function display 

#Input features should be considered  & code for what should happen when predict is clicked

   # Prediction route - handles form submission
@flask_app.route("/predict", methods=["POST"])
def predict():
    # --- Categorical feature ---
    machine_type = request.form["type"]  # Keep as string

    # --- Numeric features ---
    air_temp = float(request.form.get("air_temp", 300))
    process_temp = float(request.form.get("process_temp", 310))
    rot_speed = float(request.form.get("rot_speed", 1500))
    torque = float(request.form.get("torque", 40))
    tool_wear = float(request.form.get("tool_wear", 50))

    twf = int(request.form.get("twf", 0))
    hdf = int(request.form.get("hdf", 0))
    pwf = int(request.form.get("pwf", 0))
    osf = int(request.form.get("osf", 0))
    rnf = int(request.form.get("rnf", 0))

    # --- Derived feature ---
    temp_diff = process_temp - air_temp

    # --- Create DataFrame ---
    input_data = pd.DataFrame({
        "Type": [machine_type],           # categorical as string
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

    # --- Preprocess & Predict ---
    processed_data = pre.transform(input_data)
    prediction = model.predict(processed_data)[0]

    # --- Result message ---
    if prediction == 1:
        result_text = "Likely a Machine Failure"
    else:
        result_text = "Machine is Operating Normally"

    # Send result back to HTML
    return render_template("machine_failure.html", result=result_text)

# -------------------------
# Run Flask App
# -------------------------
if __name__ == "__main__":
    flask_app.run(debug=True)