import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os

# Load models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
preprocessor = joblib.load(os.path.join(BASE_DIR, "models", "preprocessor_v2.pkl"))
model = joblib.load(os.path.join(BASE_DIR, "models", "xgb_optuna.pkl"))
best_threshold = joblib.load(os.path.join(BASE_DIR, "models", "best_threshold_optuna.pkl"))

print(f" Threshold: {best_threshold}")

def predict_fraud(category, amt, gender, state, lat, long,
                  city_pop, merch_lat, merch_long, day_of_week,
                  month, hour, age, distance_km):

    weekend = 1 if day_of_week >= 5 else 0
    night = 1 if 0 <= hour <= 5 else 0
    log_amt = np.log1p(amt)

    if age <= 25: age_group = "0-25"
    elif age <= 40: age_group = "26-40"
    elif age <= 60: age_group = "41-60"
    else: age_group = "60+"

    if amt < 50: amount_group = "Low"
    elif amt < 200: amount_group = "Medium"
    elif amt < 1000: amount_group = "High"
    else: amount_group = "Very High"

    if city_pop < 10000: city_pop_group = "small"
    elif city_pop < 100000: city_pop_group = "medium"
    elif city_pop < 500000: city_pop_group = "large"
    else: city_pop_group = "very large"

    if distance_km < 10: distance_group = "Nearby"
    elif distance_km < 50: distance_group = "Medium"
    elif distance_km < 150: distance_group = "Far"
    else: distance_group = "Very Far"

    input_data = pd.DataFrame([{
        "category": category, "amt": amt, "gender": gender, "state": state,
        "lat": lat, "long": long, "city_pop": city_pop,
        "merch_lat": merch_lat, "merch_long": merch_long,
        "day_of_week": day_of_week, "month": month, "hour": hour,
        "weekend": weekend, "night": night, "age": age,
        "age_group": age_group, "log_amt": log_amt,
        "amount_group": amount_group, "city_pop_group": city_pop_group,
        "distance_km": distance_km, "distance_group": distance_group
    }])

    processed = preprocessor.transform(input_data)
    prob = model.predict_proba(processed)[:, 1][0]
    print(f"DEBUG → Prob: {prob:.4f} | Threshold: {best_threshold}")

    prediction = 1 if prob >= best_threshold else 0

    if prob >= 0.7:
        alert = " High Risk"
        action = "Block transaction or manual review"
    elif prob >= 0.5:
        alert = " Medium Risk"
        action = "Require OTP / verification"
    elif prob >= best_threshold:
        alert = " Low Risk"
        action = "Monitor closely"
    else:
        alert = " Normal"
        action = "Allow transaction"

    result = " FRAUD DETECTED" if prediction == 1 else " GENUINE TRANSACTION"
    return result, f"{prob:.1%}", alert, action

# Gradio UI
demo = gr.Interface(
    fn=predict_fraud,
    inputs=[
        gr.Dropdown(
            ["gas_transport","grocery_pos","shopping_net","shopping_pos",
             "food_dining","entertainment","personal_care","health_fitness",
             "misc_net","misc_pos","travel","kids_pets","home","grocery_net"],
            label="Transaction Category"
        ),
        gr.Number(label="Transaction Amount (₹)", value=100.0),
        gr.Radio(["M", "F"], label="Gender"),
        gr.Dropdown(
            ["CA","TX","NY","FL","PA","OH","IL","GA","NC","MI",
             "VA","WA","AZ","MA","TN","IN","MO","MD","WI","MN",
             "CO","AL","SC","LA","KY","OR","OK","CT","IA","MS",
             "AR","KS","UT","NV","NM","NE","WV","ID","HI","NH",
             "ME","MT","RI","DE","SD","ND","AK","VT","WY","DC"],
            label="State"
        ),
        gr.Number(label="Customer Latitude", value=40.0),
        gr.Number(label="Customer Longitude", value=-75.0),
        gr.Number(label="City Population", value=10000),
        gr.Number(label="Merchant Latitude", value=40.5),
        gr.Number(label="Merchant Longitude", value=-75.5),
        gr.Slider(0, 6, step=1, label="Day of Week (0=Mon, 6=Sun)"),
        gr.Slider(1, 12, step=1, label="Month"),
        gr.Slider(0, 23, step=1, label="Hour of Day"),
        gr.Number(label="Customer Age", value=35),
        gr.Number(label="Distance (KM)", value=10.0),
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Fraud Probability"),
        gr.Textbox(label="Alert Priority"),
        gr.Textbox(label="Recommended Action"),
    ],
    title=" Credit Card Fraud Detection",
    description="Enter transaction details to predict fraud probability"
)

if __name__ == "__main__":
    demo.launch()