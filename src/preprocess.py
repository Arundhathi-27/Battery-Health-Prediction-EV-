import pandas as pd
import numpy as np

def load_and_clean(path="data/battery_dataset.csv"):
    df = pd.read_csv(path)
    df = df.dropna()
    df = df.sort_values("cycle").reset_index(drop=True)
    return df

def compute_soh(df):
    df["SOH"] = (df["capacity"] / df["rated_capacity"]) * 100
    df["SOH"] = df["SOH"].clip(0, 100)
    return df

def compute_rul(df, eol_threshold=80):
    eol_cycle = df[df["SOH"] < eol_threshold]["cycle"].min()
    if pd.isna(eol_cycle):
        eol_cycle = df["cycle"].max()
    df["RUL"] = eol_cycle - df["cycle"]
    df["RUL"] = df["RUL"].clip(lower=0)
    return df

def label_status(soh):
    if soh >= 80:
        return "Healthy"
    elif soh >= 60:
        return "Degrading"
    else:
        return "Critical"

def add_status(df):
    df["Status"] = df["SOH"].apply(label_status)
    return df

def full_pipeline(path="data/battery_dataset.csv", save_path="data/processed_dataset.csv"):
    df = load_and_clean(path)
    df = compute_soh(df)
    df = compute_rul(df)
    df = add_status(df)
    df.to_csv(save_path, index=False)
    print(f"Processed dataset saved to {save_path}")
    return df

if __name__ == "__main__":
    full_pipeline()