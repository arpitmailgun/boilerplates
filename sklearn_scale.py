### Standardize numeric columns so each has mean 0 and standard deviation 1.

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

employees = pd.read_csv("data/employees.csv")
for column in ["salary", "years"]:
    employees[column] = employees[column].fillna(employees[column].median())

scaler = StandardScaler()
scaled = scaler.fit_transform(employees[["salary", "years"]])
result = pd.DataFrame(scaled, columns=["salary_scaled", "years_scaled"]).round(6)
result.insert(0, "name", employees["name"].to_numpy())

Path("output").mkdir(exist_ok=True)
result.to_csv("output/employees_scaled.csv", index=False)
