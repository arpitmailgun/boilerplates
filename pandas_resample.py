### Roll monthly revenue up to calendar quarters.

import pandas as pd
from pathlib import Path

revenue = pd.read_csv("data/monthly_revenue.csv", parse_dates=["month"]).sort_values("month")
quarterly = revenue.set_index("month").resample("QE").sum()
quarterly = quarterly.reset_index()
quarterly["quarter"] = quarterly["month"].dt.to_period("Q").astype(str)
quarterly = quarterly[["quarter", "revenue"]]

Path("output").mkdir(exist_ok=True)
quarterly.to_csv("output/revenue_quarterly.csv", index=False)
