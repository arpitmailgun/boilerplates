### Split a comma-separated column into one row per value.

import pandas as pd
from pathlib import Path

sample_data = pd.read_csv("data/sample_data.csv")

# "Python,Pandas,SQL" becomes a list, then one row per technology.
long = sample_data.assign(tech=sample_data.techs.str.split(",")).explode("tech")
long = long[["name", "tech"]].reset_index(drop=True)

Path("output").mkdir(exist_ok=True)
long.to_csv("output/techs_long.csv", index=False)
