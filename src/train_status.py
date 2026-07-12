import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

def train_status_model(data_path="data/processed_dataset.csv"):
    df = pd.read_csv(data_path)

    features = ["cycle", "voltage", "current", "temperature", "discharge_time", "SOH"]
    X = df[features]

    le = LabelEncoder()
    y = le.fit_transform(df["Status"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Status Model Performance:")
    print(classification_report(y_test, preds, target_names=le.classes_))

    joblib.dump(model, "models/status_model.pkl")
    joblib.dump(le, "models/status_label_encoder.pkl")
    print("Saved model to models/status_model.pkl")

if __name__ == "__main__":
    train_status_model()