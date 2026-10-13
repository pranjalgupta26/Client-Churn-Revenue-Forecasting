Project Overview
Customer churn occurs when a customer stops using a company's service. Identifying customers who may leave can help businesses prioritize retention efforts and improve customer experience.
This project creates a sample dataset for 200 enterprise clients and uses machine learning to:
- Explore customer attributes and data quality.
- Handle missing monthly-fee values.
- Train a Random Forest classifier to estimate churn.
- Evaluate classification performance.
- Rank features by their model-reported importance.
- Assign clients to low-, medium-, and high-risk categories.
- Build an illustrative 12-month revenue forecast using a Random Forest regressor.
- Plot churn-related feature importance and historical-versus-forecast revenue.
Objectives
1. Simulate client-level data for analysis.
2. Examine missing values and prepare data for modeling.
3. Predict whether a client is likely to churn.
4. Identify clients with the highest predicted churn risk.
5. Explore which input features contribute most to the trained model's predictions.
6. Forecast revenue for the next 12 months using simulated historical revenue.
7. Turn the analysis into practical business recommendations.
Tools and Technologies
- Python
- Pandas — data manipulation and tabular analysis
- NumPy — synthetic data generation and numerical operations
- Matplotlib — visualizations
- Scikit-learn — train/test split, imputation, Random Forest models, and evaluation metrics
- Google Colab — notebook execution environment
Dataset
The notebook generates data for 200 synthetic enterprise clients. It does not load real customer records.
Feature	Description
Client_ID	Unique identifier assigned to each client
Monthly_Fee	Simulated monthly client fee; some values are intentionally missing
Support_Tickets	Number of support tickets
Contract_Months	Contract duration in months
Satisfaction	Simulated customer satisfaction score
Service_Usage	Simulated service-usage value
Churn	Simulated target label: 1 indicates churn and 0 indicates no churn
Churn_Risk	Model-estimated probability of churn
Risk_Category	Risk band derived from the estimated probability


The notebook intentionally introduces missing values in approximately 12% of Monthly_Fee records. A median imputer is fitted on the training data and then applied to the test data and all clients.
Revenue data
The revenue forecasting section generates 24 months of synthetic historical revenue and predicts the following 12 months. These values are simulated, not actual financial records.
Project Workflow
1. Generate synthetic client data — create client attributes and illustrative churn labels.
2. Inspect data quality — display sample records and count missing monthly-fee values.
3. Prepare features and target — use Monthly_Fee, Support_Tickets, Contract_Months, Satisfaction, and Service_Usage as model features; use Churn as the target.
4. Split the data — use a 75% training / 25% testing split with a fixed random seed and stratification.
5. Handle missing values — replace missing feature values with training-set medians.
6. Train the churn classifier — fit a RandomForestClassifier with 100 trees and balanced class weights.
7. Evaluate the classifier — report precision, recall, F1-score, accuracy, and a confusion matrix.
8. Rank churn-related features — inspect the Random Forest feature-importance scores.
9. Score all clients — calculate churn probabilities and categorize risk as Low, Medium, or High.
10. Forecast revenue — fit a RandomForestRegressor on 24 simulated monthly observations and predict months 25–36.
11. Visualize results — plot feature importance and historical-versus-predicted revenue.
12. Generate recommendations — suggest retention and monitoring actions based on the illustrative analysis.
Model Evaluation and Illustrative Results
The notebook output shown in the project screenshots reports the following results for the generated sample:
- Synthetic clients: 200
- Missing monthly-fee records: 24 (12%)
- Simulated churn rate: 28.5%
- Test-set size: 50 clients
- Reported test accuracy: 70%
Classification report
Class	Precision	Recall	F1-score	Support
No churn (0)	0.73	0.92	0.81	36
Churn (1)	0.40	0.14	0.21	14
Macro average	0.57	0.53	0.51	50
Weighted average	0.64	0.70	0.65	50


Interpretation: The displayed model performs better at identifying non-churning clients than clients who churn. In particular, the churn-class recall is 0.14, meaning it identifies only a small proportion of actual churn cases in this test set. Accuracy alone therefore gives an incomplete picture. Improving churn recall should be a priority before using a model like this for real retention decisions.
Because the data and labels are synthetic, these metrics are illustrative and may change if the data-generation logic or random seed changes.
Feature Importance
The screenshot of the model output shows the following Random Forest feature-importance scores:
Feature	Importance
Contract_Months	0.239565
Support_Tickets	0.234407
Satisfaction	0.204758
Monthly_Fee	0.162428
Service_Usage	0.158841


In this run, contract duration and support-ticket count have the highest reported importance values among the model inputs.
Feature importance describes how the fitted model uses features; it does not establish causation. Since the labels and inputs are simulated, these rankings should not be treated as validated real-world churn drivers.

Churn Risk Categories
The notebook converts predicted churn probabilities into categories using these thresholds:
- Low: probability below 40%
- Medium: probability from 40% to below 70%
- High: probability of 70% or higher
The notebook prints the 10 clients with the highest estimated churn risk. These are model predictions on synthetic records, not real customer assessments.
Revenue Forecast
The revenue section:
- Generates 24 months of illustrative historical revenue.
- Trains a RandomForestRegressor using month number as its input.
- Predicts revenue for the next 12 months.
- Plots the historical values and forecast together.
The screenshot shows a total 12-month forecast of approximately $8,099,724.48, with the model predicting the same monthly value of approximately $674,977.04 for each of the 12 forecast months.
Forecast limitation: The model uses month number as its only predictor and is trained on a small, synthetic series. A Random Forest does not naturally extrapolate a time trend beyond the values it has seen, which can result in flat forecasts. This output should be treated as a demonstration of the workflow—not as a reliable financial forecast. A production forecast should use validated time-series data, meaningful business drivers, chronological backtesting, and suitable forecasting methods.
Visualizations
The notebook produces two plots:
1. Factors Associated with Predicted Client Churn — a horizontal bar chart of model feature importance.
2. 12-Month Revenue Forecast — a line plot comparing synthetic historical revenue with predicted future revenue.
Business Recommendations
Based on the illustrative workflow, a business could consider the following actions after validating the model on real data:
1. Review high-risk clients and investigate their service experience.
2. Examine repeated support tickets and recurring support issues.
3. Monitor customer satisfaction and follow up on negative feedback.
4. Review contract renewal timelines and consider appropriate retention plans.
5. Track churn and revenue trends regularly.
6. Measure whether retention initiatives improve outcomes using real, appropriately evaluated data.
These are general suggestions, not conclusions established by the synthetic dataset.
How to Run the Project
Option 1: Google Colab
1. Open the project notebook in Google Colab.
2. Ensure the runtime is connected.
3. Run the notebook cells from top to bottom, or select Runtime → Run all.
4. Review the printed data-quality summary, classification report, confusion matrix, high-risk client list, revenue forecast, and plots.
Option 2: Run locally
1. Install Python 3.9 or later.
2. Install the required libraries:
   pip install pandas numpy matplotlib scikit-learn
3. Save or download the notebook code as a Python file (for example, client_churn_revenue_forecasting.py) or save it as a Jupyter notebook.
4. Run the script or notebook in your Python environment.
The code generates its own synthetic data, so no external dataset is required.
Repository Structure
Client-Churn-Revenue-Forecasting/
├── README.md
└── client_churn_revenue_forecasting.py  # Suggested filename for the notebook code
The repository structure above is a suggested layout. Use the actual filename present in your repository if it differs.
Limitations and Future Improvements
- Replace synthetic records with authorized, representative company data.
- Improve churn-class recall and evaluate precision-recall trade-offs.
- Use cross-validation and tune model hyperparameters.
- Compare Random Forest with baseline models and other classifiers.
- Add calibration and choose risk thresholds based on business costs.
- Prevent data leakage and validate the full preprocessing pipeline.
- Use chronological train/test splits and time-series backtesting for revenue.
- Include relevant revenue drivers, seasonality, pricing changes, and uncertainty intervals.
- Monitor model performance and data drift after deployment.
- Protect customer data and follow applicable privacy and security requirements.
Conclusion
This project demonstrates an end-to-end introductory machine learning workflow for client churn analysis and revenue forecasting: synthetic data generation, data-quality checks, imputation, classification, model evaluation, risk scoring, regression forecasting, visualization, and business recommendations.
The most important practical lesson from the sample output is that a model can report 70% accuracy while still missing many churn cases. Evaluation should therefore consider class-specific recall and the business impact of false negatives, not accuracy alone.
