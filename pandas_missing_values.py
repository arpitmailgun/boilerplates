### Fill missing values. Numbers use the median, text uses the most common value.

import pandas as pd
from pathlib import Path

employees = pd.read_csv("data/employees.csv")

for column in ["salary", "years"]:
    employees[column] = employees[column].fillna(employees[column].median())

# mode() can return more than one value; the first is enough for this fill.
employees["role"] = employees["role"].fillna(employees["role"].mode().iloc[0])

Path("output").mkdir(exist_ok=True)
employees.to_csv("output/employees_filled.csv", index=False)
