"""
==============================================================================
 SOLAR ENERGY PREDICTION MODEL - TRAINING SCRIPT
==============================================================================
Final Year Project: Machine Learning-Based Energy Prediction and Adaptive
Power Scheduling for a Solar-Powered IoT Drainage Water Level Monitoring
Device 

Author: Adams Faisal Omokugbo (190403083)
 
WHAT THIS SCRIPT DOES
----------------------
Trains a simple linear regression model that predicts how much solar energy
the device is likely to harvest in the near future, using three things it
can measure right now: light intensity, hour of day, and battery voltage.

The model takes the form (matches Section 3.4.2, Equation 12 of the report):

    E_pred = w0 + (w1 * light) + (w2 * hour) + (w3 * battery_voltage)

Training finds the best values for w0, w1, w2, w3 by comparing predictions
against what the CT103 current sensor actually measured (the "ground truth"
solar power, calibrated by Truth).

HOW TO USE THIS SCRIPT
------------------------
1. Export the real data logged from your device from Ubidots as a CSV
   (Data -> Data Export), and make sure the columns are named:
       timestamp, light_raw, battery_voltage, solar_power
   (solar_power should come from your CT103 readings)

2. Place the CSV at TRAINING_DATA_FILE and run the script.

3. The script will print your four real weights. Copy them into the
   ESP32 firmware exactly where shown at the bottom of the output.
==============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURATION
# ==============================================================================
TRAINING_DATA_FILE = "training-data.csv"

OUTPUT_PLOT_PATH = "energy-prediction-results.png"
# ==============================================================================
# STEP 1: LOAD THE REAL TRAINING DATA
# ==============================================================================
print(f"Loading real training data from: {TRAINING_DATA_FILE}")
df = pd.read_csv(TRAINING_DATA_FILE)

if "hour" not in df.columns and "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0

if "day" not in df.columns:
    if "timestamp" in df.columns:
        df["day"] = df["timestamp"].dt.date.factorize()[0]
    else:
        df["day"] = 0

print(f"\nLoaded {len(df)} readings.")
print(df.head(), "\n")


# ==============================================================================
# STEP 2: TRAIN THE MODEL
# ==============================================================================
FEATURES = ["light_raw", "hour", "battery_voltage"]
TARGET = "solar_power"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

w0 = model.intercept_
w1, w2, w3 = model.coef_

# ==============================================================================
# STEP 3: EVALUATE HOW GOOD THE MODEL IS
# ==============================================================================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("=" * 70)
print("MODEL PERFORMANCE (on held-out test data)")
print("=" * 70)
print(f"  Mean Absolute Error (MAE):  {mae:.4f} W")
print(f"  Root Mean Squared Error:    {rmse:.4f} W")
print(f"  R-squared (fit quality):    {r2:.4f}  (1.0 = perfect, 0.0 = useless)")
print()
print("  Plain meaning: on average, the model's energy prediction is off")
print(f"  by about {mae:.3f} Watts from what actually happened.")


# ==============================================================================
# STEP 4: PRINT THE TRAINED WEIGHTS
# ==============================================================================
print("\n" + "=" * 70)
print("TRAINED MODEL WEIGHTS")
print("=" * 70)
print(f"  w0 (intercept) = {w0:.6f}")
print(f"  w1 (light)     = {w1:.8f}")
print(f"  w2 (hour)      = {w2:.6f}")
print(f"  w3 (battery)   = {w3:.6f}")
print()
print("  Formula:")
print(f"  E_pred = {w0:.6f} + ({w1:.8f} * light) + ({w2:.6f} * hour) + ({w3:.6f} * battery_voltage)")


# ==============================================================================
# STEP 5: GENERATE THE ESP32 FIRMWARE SNIPPET, READY TO PASTE
# ==============================================================================
firmware_snippet = f"""
// ============================================================
// ENERGY PREDICTION MODEL WEIGHTS - trained {pd.Timestamp.now().strftime('%Y-%m-%d')}
// Paste this into your ESP32 firmware
// ============================================================
#define MODEL_W0 {w0:.6f}f
#define MODEL_W1 {w1:.8f}f
#define MODEL_W2 {w2:.6f}f
#define MODEL_W3 {w3:.6f}f

float predictEnergy(float lightRaw, float hourOfDay, float batteryVoltage) {{
    float ePred = MODEL_W0
                + (MODEL_W1 * lightRaw)
                + (MODEL_W2 * hourOfDay)
                + (MODEL_W3 * batteryVoltage);
    if (ePred < 0) ePred = 0;   // energy prediction can't be negative
    return ePred;
}}
"""

print("\n" + "=" * 70)
print("ESP32 FIRMWARE CODE - COPY THIS INTO YOUR PROJECT")
print("=" * 70)
print(firmware_snippet)

with open("esp32_energy_prediction_snippet.txt", "w") as f:
    f.write(firmware_snippet)
print("(Also saved to esp32_energy_prediction_snippet.txt)")


# ==============================================================================
# STEP 6: PLOT PREDICTED VS ACTUAL (good for Chapter 4)
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: predicted vs actual scatter
axes[0].scatter(y_test, y_pred, alpha=0.5, s=15, color="#2b6cb0")
lims = [0, max(y_test.max(), y_pred.max()) * 1.05]
axes[0].plot(lims, lims, 'r--', linewidth=1.5, label="Perfect prediction")
axes[0].set_xlabel("Actual Solar Power (W)")
axes[0].set_ylabel("Predicted Solar Power (W)")
axes[0].set_title(f"Predicted vs Actual (R² = {r2:.3f})")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Right: one day's worth, predicted vs actual over time
day0 = df[df["day"] == 0].sort_values("hour")
day0_pred = model.predict(day0[FEATURES])
axes[1].plot(day0["hour"], day0["solar_power"], label="Actual", color="#2b6cb0", linewidth=2)
axes[1].plot(day0["hour"], day0_pred, label="Predicted", color="#e53e3e", linewidth=2, linestyle="--")
axes[1].set_xlabel("Hour of Day")
axes[1].set_ylabel("Solar Power (W)")
axes[1].set_title("Example Day: Predicted vs Actual Energy")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT_PATH, dpi=150)
print(f"\nPlot saved to: {OUTPUT_PLOT_PATH}")
print("\nDONE.")
