### Turn category columns into 0/1 columns.

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder

employees = pd.read_csv("data/employees.csv")
employees["role"] = employees["role"].fillna(employees["role"].mode().iloc[0])

encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(employees[["department", "role"]])
columns = encoder.get_feature_names_out(["department", "role"])
result = pd.DataFrame(encoded, columns=columns).astype(int)
result.insert(0, "name", employees["name"].to_numpy())

Path("output").mkdir(exist_ok=True)
result.to_csv("output/employees_one_hot.csv", index=False)
