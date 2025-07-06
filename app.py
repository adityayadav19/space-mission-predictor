import gradio as gr
import pickle
import numpy as np
import os
import gdown

# 🔽 Google Drive model download
FILE_ID = "18ZOqqhbHA9YGlrV85MzrmsVPkZXgcY14"
URL = f"https://drive.google.com/uc?id={FILE_ID}"
MODEL_PATH = "regressor.pkl"

if not os.path.exists(MODEL_PATH):
    print("🔽 Downloading model using gdown...")
    gdown.download(URL, MODEL_PATH, quiet=False)
    print("✅ Model downloaded!")

with open(MODEL_PATH, 'rb') as f:
    regressor = pickle.load(f)

MISSION_NAME_MAP = {f"Mission-{i}": i-1 for i in range(1, 501)}
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
}
MISSION_TYPE_MAP = {
    "Exploration": 0,
    "Research": 1,
    "Colonization": 2,
    "Mining": 3,
}
LAUNCH_VEHICLE_MAP = {
    "SLS": 0,
    "Starship": 1,
    "Ariane 6": 2,
    "Falcon Heavy": 3,
}

def predict_success(mission_name, target_type, target_name, mission_type, launch_vehicle,
                    distance, duration, cost, yield_score, crew_size, fuel, payload):
    try:
        input_features = [
            MISSION_NAME_MAP[mission_name],
            TARGET_TYPE_MAP[target_type],
            TARGET_NAME_MAP[target_name],
            MISSION_TYPE_MAP[mission_type],
            LAUNCH_VEHICLE_MAP[launch_vehicle],
            float(distance),
            float(duration),
            float(cost),
            float(yield_score),
            float(crew_size),
            float(fuel),
            float(payload)
        ]
        prediction = regressor.predict([input_features])[0]
        return f"Predicted Mission Success Rate: {round(prediction, 2)}%"
    except Exception as e:
        return f"Error: {str(e)}"

iface = gr.Interface(
    fn=predict_success,
    inputs=[
        gr.Dropdown(list(MISSION_NAME_MAP.keys()), label="Mission Name"),
        gr.Dropdown(list(TARGET_TYPE_MAP.keys()), label="Target Type"),
        gr.Dropdown(list(TARGET_NAME_MAP.keys()), label="Target Name"),
        gr.Dropdown(list(MISSION_TYPE_MAP.keys()), label="Mission Type"),
        gr.Dropdown(list(LAUNCH_VEHICLE_MAP.keys()), label="Launch Vehicle"),
        gr.Number(label="Distance from Earth (light-years)"),
        gr.Number(label="Mission Duration (years)"),
        gr.Number(label="Mission Cost (billion USD)"),
        gr.Number(label="Scientific Yield (points)"),
        gr.Number(label="Crew Size"),
        gr.Number(label="Fuel Consumption (tons)"),
        gr.Number(label="Payload Weight (tons)")
    ],
    outputs="text",
    title="🚀 Space Mission Success Predictor",
    description="Input mission parameters and get the predicted success rate."
)

iface.launch()
