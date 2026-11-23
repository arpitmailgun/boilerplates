### Score a classifier on several held-out folds instead of one split.

import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

customers = pd.read_csv("data/churn.csv")
features = ["monthly_spend", "support_tickets", "tenure_months"]
model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, random_state=0)),
    ]
)
folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
scores = cross_val_score(
    model,
    customers[features],
    customers["churned"],
    cv=folds,
    scoring="accuracy",
)

fold_scores = pd.DataFrame({"fold": range(1, len(scores) + 1), "accuracy": scores.round(4)})
summary = pd.DataFrame(
    [{"mean_accuracy": round(float(scores.mean()), 4), "folds": len(scores)}]
)

Path("output").mkdir(exist_ok=True)
fold_scores.to_csv("output/churn_cv_folds.csv", index=False)
summary.to_csv("output/churn_cv_summary.csv", index=False)
