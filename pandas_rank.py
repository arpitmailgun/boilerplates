### Rank salary inside each department. The lowest salary is rank 1.

import pandas as pd
from pathlib import Path

employees = pd.read_csv("data/employees.csv")
employees["salary_rank"] = employees.groupby("department")["salary"].rank(method="min", ascending=True)

Path("output").mkdir(exist_ok=True)
employees.to_csv("output/salary_rank.csv", index=False)
