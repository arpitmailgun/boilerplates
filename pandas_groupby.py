### Summarize rows by a category. Mean salary, mean years, and headcount.

import pandas as pd
from pathlib import Path

employees = pd.read_csv("data/employees.csv")

summary = employees.groupby("department", as_index=False).agg(
    people=("name", "size"),
    mean_salary=("salary", "mean"),
    mean_years=("years", "mean"),
)
summary = summary.sort_values("department")
summary["mean_salary"] = summary["mean_salary"].round(2)
summary["mean_years"] = summary["mean_years"].round(2)

Path("output").mkdir(exist_ok=True)
summary.to_csv("output/department_summary.csv", index=False)
