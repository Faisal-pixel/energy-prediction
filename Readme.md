# Solar Energy Prediction

This project trains a machine-learning model to predict the solar power
available to an ESP32-based IoT drainage water-level monitoring device.

The model uses real sensor readings from the device:

- `light_raw`: raw light-intensity sensor reading
- `hour`: hour of the day as a decimal value
- `battery_voltage`: measured battery voltage

The target value is:

- `solar_power`: solar power measured by the CT103 current sensor, in watts

The model is a multiple linear regression model:

```text
E_pred = w0 + (w1 * light_raw) + (w2 * hour) + (w3 * battery_voltage)
```

## Project files

| File | Description |
| --- | --- |
| `energy-prediction-model-training.py` | Loads the real data, trains and evaluates the model, prints the model weights, and generates the ESP32 code snippet |
| `training-data.csv` | Real sensor data used for training |
| `energy-prediction-results.png` | Generated comparison plot |
| `esp32_energy_prediction_snippet.txt` | Generated C/C++ firmware code containing the trained weights |

The last two files are generated when the training script runs and do not
need to be committed to GitHub unless they are required as project results.

## Requirements

- Python 3.10 or newer
- NumPy
- pandas
- scikit-learn
- Matplotlib

## Setup

From the project directory, create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Training the model

Make sure `training-data.csv` is in the same directory as the training
script, then run:

```bash
python energy-prediction-model-training.py
```

The script will:

1. Load the real readings from `training-data.csv`.
2. Use `hour` and `day` from the CSV when they are available.
3. Calculate `hour` and `day` from `timestamp` when those columns are
   available instead.
4. Split the data into training and test sets.
5. Train a linear regression model.
6. Print MAE, RMSE, and R-squared evaluation results.
7. Print the trained model weights.
8. Generate the ESP32 firmware snippet.
9. Save a predicted-versus-actual plot.

The script uses an 80/20 train/test split with a fixed random state of `42`
so that results are reproducible.

## CSV format

The training CSV must contain these columns:

```csv
day,hour,light_raw,battery_voltage,solar_power
0,0.0,17.4,3.749,0.0
0,0.08333333333333333,21.0,3.749,0.0687
```

For a Ubidots export that contains timestamps, the minimum required columns
are:

```csv
timestamp,light_raw,battery_voltage,solar_power
2026-08-01 00:00:00,17.4,3.749,0.0
```

The script derives `hour` from `timestamp` if `hour` is not already present.
It also derives a day index when `day` is not present.

## Using the trained weights

After training, copy the generated definitions from
`esp32_energy_prediction_snippet.txt` into the ESP32 firmware:

```cpp
#define MODEL_W0 ...
#define MODEL_W1 ...
#define MODEL_W2 ...
#define MODEL_W3 ...
```

The generated `predictEnergy()` function accepts the current light reading,
hour of day, and battery voltage, and returns the predicted solar power. Any
negative prediction is limited to zero.

## Data and repository hygiene

Only use real device measurements for the final model. Do not mix the
previous practice data with the real training data unless that approach has
been deliberately evaluated and justified.

Do not commit local virtual environments such as `venv/` or `.venv/`.
Instead, commit a dependency file such as `requirements.txt` if the project
needs a reproducible installation setup.

Before committing `training-data.csv`, check that it does not contain
credentials, private identifiers, device tokens, or other sensitive
information. If the measurements are confidential, keep the CSV local and
document the required column format instead.

## Project context

This model supports the final-year project:

**Machine Learning-Based Energy Prediction and Adaptive Power Scheduling for
a Solar-Powered IoT Drainage Water Level Monitoring Device**

