### Compare each salary with the mean of its own department.

import pandas as pd
from pathlib import Path

employees = pd.read_csv("data/employees.csv")
# transform keeps one result per row, aligned with the original index.
department_mean = employees.groupby("department")["salary"].transform("mean")
employees["department_mean_salary"] = department_mean.round(2)
employees["salary_share"] = (employees["salary"] / department_mean).round(4)

Path("output").mkdir(exist_ok=True)
employees.to_csv("output/salary_share.csv", index=False)
