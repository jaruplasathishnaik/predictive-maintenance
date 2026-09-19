# AI-Powered Predictive Maintenance System

An end-to-end Machine Learning web application designed to predict equipment failures before they happen. Built using *Random Forest* and deployed with a *Streamlit* dashboard interface.

## 🛠️ Repository Architecture

* `app.py` - The user-friendly frontend dashboard interface built with Streamlit.
* `model.py` - Core machine learning logic containing the training workflow and predictions.
* `predictive_maintenance.csv` - The historical telemetry sensor dataset used for training the model.
* `rf_model.pkl` - Saved production-ready Random Forest classifier model weights.
* `scaler.pkl` - Serialized normalization/scaling metrics mapping input features uniformly.

## 🚀 Key Features

* **Failure Prediction Engine:** Classifies machine telemetry inputs into healthy or critical failure modes.
* **Real-time Inference Dashboard:** Streamlined data entry widgets for continuous parameter inspection.
* **Pre-processed Scaling pipelines:** Built-in safeguards preventing mathematical data leakage during standard predictions.

## 🔧 Installation & Quickstart

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd predictive-maintenance
   ```

2. **Install necessary dependencies:**
   ```bash
   pip install streamlit pandas scikit-learn
   ```

3. **Launch the app UI natively:**
   ```bash
   git clone https://github.com
   ```
