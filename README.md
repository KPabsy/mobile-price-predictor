# Mobile Phone Price Predictor 📱💰

A Deep Learning project that classifies mobile phones into price ranges based on their hardware specifications (RAM, Battery Power, Processor speed, etc.). This model uses an Artificial Neural Network (ANN) to achieve high classification accuracy.

## 🚀 Project Overview
While many price predictors use regression to find an exact cost, this project treats the problem as a **Binary Classification** task to determine the price category of a device.

* **Model Type:** Artificial Neural Network (ANN)
* **Framework:** Keras/TensorFlow
* **Task:** Binary Classification (Price Range)

## 🏗️ Model Architecture
The model is a Sequential Deep Learning model with the following layers:
1. **Input Layer:** Dense layer with 8 units and `ReLU` activation.
2. **Hidden Layer:** Dense layer with 4 units and `ReLU` activation.
3. **Output Layer:** Dense layer with 1 unit and `Sigmoid` activation (to output probabilities between 0 and 1).

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning:** Keras, TensorFlow
* **Data Processing:** Pandas, NumPy, Scikit-learn (StandardScaler)
* **Environment:** Google Colab

## 🧹 Data Preprocessing
To ensure the Neural Network performs optimally, the data underwent:
* **Feature Scaling:** Used `StandardScaler` to normalize the input features, ensuring all hardware specs are on the same scale for faster convergence.
* **Train-Test Split:** Divided the dataset into a 75% training set and a 25% testing set to evaluate performance.

## 📊 Training & Evaluation
* **Loss Function:** `binary_crossentropy`
* **Optimizer:** `adam`
* **Metrics:** Accuracy
* **Epochs:** 100

## 📦 Usage
To use the trained model (`weights.keras`) for your own predictions:

```python
from keras.models import load_model
import numpy as np

# Load the model
model = load_model('weights.keras')

# Example hardware specs (must be scaled first)
# prediction = model.predict(scaled_data)
# result = (prediction > 0.5).astype(int)
