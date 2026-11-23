### Measure which column the classifier actually relies on.

import pandas as pd
from pathlib import Path
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

rows = pd.read_csv("data/signal.csv")
features = ["noise", "signal"]
model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, random_state=0)),
    ]
)
x = rows[features]
y = rows["label"]
model.fit(x, y)
# Shuffle one column at a time. A drop in accuracy is that column's importance.
importance = permutation_importance(model, x, y, n_repeats=20, random_state=0, scoring="accuracy")

result = pd.DataFrame(
    {
        "feature": features,
        "importance_mean": importance.importances_mean.round(4),
        "importance_std": importance.importances_std.round(4),
    }
).sort_values("importance_mean", ascending=False)

Path("output").mkdir(exist_ok=True)
result.to_csv("output/feature_importance.csv", index=False)
