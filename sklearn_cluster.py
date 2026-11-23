### Group employees into two clusters from salary and years.

import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

employees = pd.read_csv("data/employees.csv")
for column in ["salary", "years"]:
    employees[column] = employees[column].fillna(employees[column].median())

scaler = StandardScaler()
scaled = scaler.fit_transform(employees[["salary", "years"]])
clusters = KMeans(n_clusters=2, n_init=10, random_state=0)
employees["cluster"] = clusters.fit_predict(scaled)

Path("output").mkdir(exist_ok=True)
employees.to_csv("output/employee_clusters.csv", index=False)
