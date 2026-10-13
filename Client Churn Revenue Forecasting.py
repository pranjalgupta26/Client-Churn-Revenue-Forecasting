# Client Churn Risk & Revenue Forecasting
# CountryEdu Assignment - Question 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score
)

np.random.seed(42)

# STEP 1: Generate synthetic data for 200 enterprise clients

n_clients = 200

clients = pd.DataFrame({
    "Client_ID": range(1, n_clients + 1),
    "Monthly_Fee": np.random.randint(2000, 15000, n_clients),
    "Support_Tickets": np.random.randint(1, 30, n_clients),
    "Contract_Months": np.random.randint(1, 61, n_clients),
    "Satisfaction": np.round(
        np.random.uniform(1, 5, n_clients), 1
    ),
    "Service_Usage": np.random.uniform(0.2, 1.0, n_clients)
})

# STEP 2: Generate illustrative churn labels
# These labels are simulated, not actual company records.

churn_probability = (
    0.35
    + 0.015 * clients["Support_Tickets"]
    - 0.07 * clients["Satisfaction"]
    - 0.003 * clients["Contract_Months"]
)

churn_probability = np.clip(churn_probability, 0.02, 0.80)

clients["Churn"] = (
    np.random.random(n_clients) < churn_probability
).astype(int)

# STEP 3: Simulate missing financial data (12%)
missing_indices = np.random.choice(
    clients.index,
    size=int(0.12 * n_clients),
    replace=False
)

clients.loc[missing_indices, "Monthly_Fee"] = np.nan

print("=" * 55)
print("CLIENT CHURN RISK & REVENUE FORECASTING")
print("=" * 55)

print("\nSample Client Records:")
print(clients.head(10).to_string(index=False))

print("\n--- DATA QUALITY ---")
print("Total enterprise clients:", len(clients))
print(
    "Missing monthly fee records:",
    clients["Monthly_Fee"].isna().sum()
)
print(
    "Missing financial data:",
    round(clients["Monthly_Fee"].isna().mean() * 100, 2),
    "%"
)

# STEP 4: Prepare data for machine learning

features = [
    "Monthly_Fee",
    "Support_Tickets",
    "Contract_Months",
    "Satisfaction",
    "Service_Usage"
]

X = clients[features]
y = clients["Churn"]

# Split before fitting the imputer to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

imputer = SimpleImputer(strategy="median")

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# STEP 5: Train the churn prediction model

churn_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)

churn_model.fit(X_train_imputed, y_train)

churn_predictions = churn_model.predict(X_test_imputed)

print("\n--- CHURN MODEL RESULTS ---")
print("Actual simulated churn rate:",
      round(y.mean() * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        churn_predictions,
        zero_division=0
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, churn_predictions))

# STEP 6: Identify important churn factors

importance = pd.Series(
    churn_model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\n--- TOP CHURN FACTORS ---")
print(importance.to_string())

# STEP 7: Predict churn risk for all clients

all_clients_imputed = imputer.transform(X)

clients["Churn_Risk"] = churn_model.predict_proba(
    all_clients_imputed
)[:, 1]

clients["Risk_Category"] = pd.cut(
    clients["Churn_Risk"],
    bins=[-0.001, 0.40, 0.70, 1.0],
    labels=["Low", "Medium", "High"]
)

print("\n--- CLIENTS WITH HIGHEST PREDICTED CHURN RISK ---")

print(
    clients[
        ["Client_ID", "Churn_Risk", "Risk_Category"]
    ]
    .sort_values("Churn_Risk", ascending=False)
    .head(10)
    .to_string(index=False)
)

# STEP 8: Build an illustrative 12-month revenue forecast

# Simulate 24 months of historical company revenue.
# These figures are synthetic, not real financial records.

months = np.arange(24)

historical_revenue = (
    500000
    + months * 8000
    + 30000 * np.sin(2 * np.pi * months / 12)
    + np.random.normal(0, 10000, 24)
)

revenue_data = pd.DataFrame({
    "Month_Number": months,
    "Revenue": historical_revenue
})

revenue_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

revenue_model.fit(
    revenue_data[["Month_Number"]],
    revenue_data["Revenue"]
)

future_months = np.arange(24, 36)

future_revenue = revenue_model.predict(
    pd.DataFrame({"Month_Number": future_months})
)

forecast = pd.DataFrame({
    "Forecast_Month": np.arange(1, 13),
    "Predicted_Revenue": np.round(future_revenue, 2)
})

print("\n--- 12-MONTH REVENUE FORECAST ---")
print(forecast.to_string(index=False))

print(
    "\nTotal forecast revenue:",
    f"${forecast['Predicted_Revenue'].sum():,.2f}"
)

# STEP 9: Visualize churn risk and revenue forecast

plt.figure(figsize=(8, 5))
importance.sort_values().plot(kind="barh")
plt.title("Factors Associated with Predicted Client Churn")
plt.xlabel("Model Feature Importance")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(
    np.arange(1, 25),
    historical_revenue,
    marker="o",
    label="Synthetic Historical Revenue"
)
plt.plot(
    np.arange(25, 37),
    future_revenue,
    marker="o",
    linestyle="--",
    label="Predicted Revenue"
)
plt.title("12-Month Revenue Forecast")
plt.xlabel("Month")
plt.ylabel("Revenue ($)")
plt.legend()
plt.tight_layout()
plt.show()

# STEP 10: Business recommendations

print("\n--- RECOMMENDATIONS ---")
print("1. Contact high-risk clients proactively.")
print("2. Investigate repeated support tickets.")
print("3. Improve customer satisfaction through regular reviews.")
print("4. Offer suitable service improvements and renewal plans.")
print("5. Monitor churn and revenue trends monthly.")
print("6. Measure whether retention initiatives reduce churn.")

print("\n--- PROJECT COMPLETED ---")
