### Pivot mean salary so each role is a column.

import pandas as pd
from pathlib import Path

employees = pd.read_csv("data/employees.csv")
# A missing role or salary would become its own pivot column or an empty cell.
usable = employees.dropna(subset=["role", "salary"])

table = pd.pivot_table(
    usable,
    index="department",
    columns="role",
    values="salary",
    aggfunc="mean",
)
table = table.reindex(sorted(table.columns), axis=1).sort_index().round(2)

Path("output").mkdir(exist_ok=True)
table.to_csv("output/salary_pivot.csv")
