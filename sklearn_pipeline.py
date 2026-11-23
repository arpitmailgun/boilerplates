### Impute, scale, one-hot encode, and classify in one pipeline.

import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

customers = pd.read_csv("data/customers.csv")
numeric = ["monthly_spend", "support_tickets", "tenure_months"]
categorical = ["plan"]

prep = ColumnTransformer(
    [
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric,
        ),
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
                ]
            ),
            categorical,
        ),
    ]
)
model = Pipeline(
    [
        ("prep", prep),
        ("clf", LogisticRegression(max_iter=200, random_state=0)),
    ]
)

x = customers[numeric + categorical]
y = customers["churned"]
x_train, x_test, y_train, y_test, names_train, names_test = train_test_split(
    x,
    y,
    customers["customer"],
    test_size=0.25,
    random_state=0,
    stratify=y,
)
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
    [
        {
            "accuracy": round(float(accuracy_score(y_test, predicted)), 4),
            "test_rows": len(y_test),
            "features": ",".join(model.named_steps["prep"].get_feature_names_out()),
        }
    ]
)

Path("output").mkdir(exist_ok=True)
predictions.to_csv("output/pipeline_predictions.csv", index=False)
metrics.to_csv("output/pipeline_metrics.csv", index=False)
