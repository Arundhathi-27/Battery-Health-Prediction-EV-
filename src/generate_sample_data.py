import numpy as np
import pandas as pd

np.random.seed(42)

n_cycles = 500
cycle = np.arange(1, n_cycles + 1)

rated_capacity = 2.0
capacity = rated_capacity * (1 - 0.0007 * cycle - 0.0000015 * cycle**2)
capacity += np.random.normal(0, 0.01, n_cycles)
capacity = np.clip(capacity, 0.5, rated_capacity)

voltage = np.random.normal(3.7, 0.05, n_cycles)
current = np.random.normal(-1.5, 0.1, n_cycles)
temperature = np.random.normal(25, 3, n_cycles)
discharge_time = np.random.normal(3600, 200, n_cycles)

df = pd.DataFrame({
    "cycle": cycle,
    "voltage": voltage,
    "current": current,
    "temperature": temperature,
    "discharge_time": discharge_time,
    "capacity": capacity,
    "rated_capacity": rated_capacity
})

df.to_csv("data/battery_dataset.csv", index=False)
print("Sample dataset created at data/battery_dataset.csv")
print(df.head())