### Search the regularization strength with cross-validation.

import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

customers = pd.read_csv("data/churn_search.csv")
features = ["monthly_spend", "support_tickets", "tenure_months"]
model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, random_state=0)),
    ]
)
search = GridSearchCV(
    model,
    param_grid={"clf__C": [0.01, 0.1, 1.0, 10.0]},
    cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=0),
    scoring="accuracy",
)
search.fit(customers[features], customers["churned"])

results = pd.DataFrame(search.cv_results_)
results = results[["param_clf__C", "mean_test_score", "rank_test_score"]].sort_values("rank_test_score")
results = results.rename(columns={"param_clf__C": "C", "mean_test_score": "mean_accuracy"})
results["mean_accuracy"] = results["mean_accuracy"].astype(float).round(4)
selected = float(search.best_params_["clf__C"])
results["selected"] = results["C"].astype(float).eq(selected)

Path("output").mkdir(exist_ok=True)
results.to_csv("output/churn_grid_search.csv", index=False)
