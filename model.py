# model.py

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


def preprocess_data(df):
    df_clean = df.drop(['UDI', 'Product ID', 'Failure Type'], axis=1)

    df_clean['Temp_Diff'] = (
        df_clean['Process temperature [K]'] -
        df_clean['Air temperature [K]']
    )

    le = LabelEncoder()
    df_clean['Type'] = le.fit_transform(df_clean['Type'])

    X = df_clean.drop('Target', axis=1)
    y = df_clean['Target']

    return X, y


def train_and_save_model(csv_path):

    print("🚀 Training started...")

    df = pd.read_csv(csv_path)

    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42  
    )

    model.fit(X_train_res, y_train_res)

    # ✅ Save model and scaler
    joblib.dump(model, "rf_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    print("✅ Model and Scaler saved successfully!")


# 🔥 THIS PART WAS MISSING
if __name__ == "__main__":
    train_and_save_model("predictive_maintenance.csv")
