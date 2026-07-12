import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def train_soh_model(data_path="data/processed_dataset.csv"):
    df = pd.read_csv(data_path)
    features = ["cycle", "voltage", "current", "temperature", "discharge_time"]
    X = df[features]
    y = df["SOH"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("SOH Model Performance:")
    print("MAE:", mean_absolute_error(y_test, preds))
    print("R2:", r2_score(y_test, preds))

    joblib.dump(model, "models/soh_model.pkl")
    print("Saved model to models/soh_model.pkl")

if __name__ == "__main__":
    train_soh_model()