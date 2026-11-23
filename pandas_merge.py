### Attach department location and budget onto each employee.

import pandas as pd
from pathlib import Path

employees = pd.read_csv("data/employees.csv")
departments = pd.read_csv("data/departments.csv")
combined = employees.merge(departments, on="department", how="left")

Path("output").mkdir(exist_ok=True)
combined.to_csv("output/employees_with_department.csv", index=False)
