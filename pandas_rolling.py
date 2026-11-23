### Smooth a monthly series with a trailing three-month mean.

import pandas as pd
from pathlib import Path

revenue = pd.read_csv("data/monthly_revenue.csv", parse_dates=["month"]).sort_values("month")
# The first two months stay empty because the window needs three observations.
revenue["rolling_3_mean"] = revenue["revenue"].rolling(window=3, min_periods=3).mean().round(2)

Path("output").mkdir(exist_ok=True)
revenue.to_csv("output/revenue_rolling.csv", index=False)
