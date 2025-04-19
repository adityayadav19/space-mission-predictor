from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained regressor
with open('regressor.pkl', 'rb') as f:
    regressor = pickle.load(f)

@app.route('/')
def index():
    print("Index route accessed")
    return render_template('index.html')
MISSION_NAME_MAP ={f"Mission-{i}": i-1 for i in range(1, 501)}
TARGET_TYPE_MAP = {
    "Planet": 0,
    "Moon": 1,
    "Asteroid": 2,
    "Star": 3,
    "Exoplanet": 4,
}
TARGET_NAME_MAP = {
    "Mars": 0,
    "Titan": 1,
    "Betelgeuse": 2,
    "Proxima b": 3,
    "Ceres": 4,
    "Europa": 5,
    "Io": 6,
    # Add all target names...
}
MISSION_TYPE_MAP = {
    "Exploration": 0,
    "Research": 1,
    "Colonization": 2,
    "Mining": 3,
    # Add all mission types...
}
LAUNCH_VEHICLE_MAP = {
    "SLS": 0,
    "Starship": 1,
    "Ariane 6": 2,
    "Falcon Heavy": 3,
}
@app.route('/predict', methods=['POST'])
def predict():
    print("Form keys received:", list(request.form.keys()))
    print("Form data received:", dict(request.form))
     # Convert categorical dropdowns to numeric using mappings
    mission_name = MISSION_NAME_MAP[request.form['mission_name']]
    target_type = TARGET_TYPE_MAP[request.form['target_type']]
    target_name = TARGET_NAME_MAP[request.form['target_name']]
    mission_type = MISSION_TYPE_MAP[request.form['mission_type']]
    launch_vehicle = LAUNCH_VEHICLE_MAP[request.form['launch_vehicle']]

    input_features = [
        mission_name,
        target_type,
        target_name,
        mission_type,
        launch_vehicle,
        float(request.form['Distance from Earth (light-years)']),
        float(request.form['Mission Duration (years)']),
        float(request.form['Mission Cost (billion USD)']),
        float(request.form['Scientific Yield (points)']),
        float(request.form['Crew Size']),
        float(request.form['Fuel Consumption (tons)']),
        float(request.form['Payload Weight (tons)']),
    ]

    prediction = regressor.predict([input_features])[0]
    prediction = round(prediction, 2)

    return render_template('result.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
