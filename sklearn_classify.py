### Train a classifier and save the held-out predictions.

import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

customers = pd.read_csv("data/churn.csv")
features = ["monthly_spend", "support_tickets", "tenure_months"]
x = customers[features]
y = customers["churned"]

x_train, x_test, y_train, y_test, names_train, names_test = train_test_split(
    x,
    y,
    customers["customer"],
    test_size=0.25,
    random_state=0,
    stratify=y,
)

model = LogisticRegression(max_iter=200, random_state=0)
model.fit(x_train, y_train)
predicted = model.predict(x_test)

predictions = pd.DataFrame(
    {
        "customer": names_test.to_numpy(),
        "actual": y_test.to_numpy(),
        "predicted": predicted,
    }
).sort_values("customer")
metrics = pd.DataFrame(
    [{"accuracy": round(float(accuracy_score(y_test, predicted)), 4), "test_rows": len(y_test)}]
)

Path("output").mkdir(exist_ok=True)
predictions.to_csv("output/churn_predictions.csv", index=False)
metrics.to_csv("output/churn_metrics.csv", index=False)
