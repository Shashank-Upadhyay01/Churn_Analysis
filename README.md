Churn Analysis and Prediction Project
This project focuses on customer churn analysis using Power BI, SQL, Excel, and Python. The goal is to explore customer data, visualize churn insights, and build a predictive model to identify potential churners.

# Tools & Technologies Used
  - Power BI: Data visualization and reporting
  - SQL: Data exploration and transformation
  - Excel: Data handling and integration
  - Python: Machine learning model for churn prediction

# SQL Queries
## Data Exploration
  1. Check Distinct Values by Gender, Contract, and Customer Status:
 ```
SELECT Gender, Count(Gender) as TotalCount, Count(Gender) * 1.0 / (Select Count(*) from stg_Churn)  as Percentage
FROM stg_Churn
GROUP BY Gender; 
SELECT Contract, Count(Contract) as TotalCount, Count(Contract) * 1.0 / (Select Count(*) from stg_Churn)  as Percentage
FROM stg_Churn
GROUP BY Contract;
SELECT Customer_Status, Count(Customer_Status) as TotalCount, Sum(Total_Revenue) as TotalRev,
Sum(Total_Revenue) / (Select sum(Total_Revenue) from stg_Churn) * 100 as RevPercentage
FROM stg_Churn
GROUP BY Customer_Status;
```
2. Checking for Null Values:
```
SELECT SUM(CASE WHEN Customer_ID IS NULL THEN 1 ELSE 0 END) AS Customer_ID_Null_Count, 
...
SUM(CASE WHEN Churn_Reason IS NULL THEN 1 ELSE 0 END) AS Churn_Reason_Null_Count
FROM stg_Churn;
```
## Data Transformation
1. Inserting Data into Production Table:
```
SELECT Customer_ID, Gender, Age, Married, State, Number_of_Referrals, Tenure_in_Months,
ISNULL(Value_Deal, 'None') AS Value_Deal, ...
INTO [db_Churn].[dbo].[prod_Churn]
FROM [db_Churn].[dbo].[stg_Churn];
```
2. Creating Views for Power BI:
```
CREATE VIEW vw_ChurnData AS
SELECT * FROM prod_Churn WHERE Customer_Status IN ('Churned', 'Stayed');

CREATE VIEW vw_JoinData AS
SELECT * FROM prod_Churn WHERE Customer_Status = 'Joined';
```
# 2. Power BI Queries
## Power Query Transformations
1. New Columns:
Churn Status:
```
Churn Status = if [Customer_Status] = "Churned" then 1 else 0
```
Monthly Charge Range:
```
Monthly Charge Range = if [Monthly_Charge] < 20 then "< 20" ...
```
2. Table References for Age and Tenure Groups:
 - Age and Tenure groups were mapped to create categories like "< 20" or "< 6 Months."

3. Unpivoting Services Columns:
  - Unpivot columns for Service Attributes like Device Protection, Streaming TV, etc.
## Measures
  - Total Customers:
```
Total Customers = Count(prod_Churn[Customer_ID])
```
  - Total Churn:
```
Total Churn = SUM(prod_Churn[Churn Status])
```
  - Churn Rate:
```
Churn Rate = [Total Churn] / [Total Customers]
```
  - Predicted Churners:
```
Title Predicted Churners = "COUNT OF PREDICTED CHURNERS : " & COUNT(Predictions[Customer_ID])
```
# 3. Python Model for Churn Prediction
## Installation of Libraries
```
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```
## Data Preprocessing
```
Import Libraries & Load Data:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
```
## Load data
```
data = pd.read_excel("Prediction_Data.xlsx", sheet_name='vw_ChurnData')
```
## Data Cleanup:
```
data = data.drop(['Customer_ID', 'Churn_Category', 'Churn_Reason'], axis=1)
columns_to_encode = ['Gender', 'Married', 'State', ...]
label_encoders = {col: LabelEncoder() for col in columns_to_encode}
for col in columns_to_encode:
    data[col] = label_encoders[col].fit_transform(data[col])
```
## Model Training
```
Training the Random Forest Classifier:
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
```
## Model Evaluation:
```
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```
## Feature Importance Visualization:
```
importances = rf_model.feature_importances_
sns.barplot(x=importances[indices], y=X.columns[indices])
```
## Prediction on New Data
```
new_data = pd.read_excel("Prediction_Data.xlsx", sheet_name='vw_JoinData')
new_data = new_data.drop(['Customer_ID', 'Customer_Status', 'Churn_Category', 'Churn_Reason'], axis=1)
new_predictions = rf_model.predict(new_data)
original_data['Customer_Status_Predicted'] = new_predictions
original_data.to_csv("Predictions.csv", index=False)
```
# How to Run the Project
 ### 1. Clone the Repository
 ### 2. SQL Queries:
  - Load and run the SQL scripts to explore and clean data in your database.
 ### 3.Power BI Dashboard:
  - Open the Power BI file and explore the churn insights.
 ### 4. Python Prediction Model:
  - Open Jupyter Notebook and execute the provided script to build and test the churn prediction model.
### 5. Prediction:
  - Run the prediction section to generate a CSV file with predicted churners.
# Future Enhancements
  - Expand the churn prediction model by exploring other algorithms like XGBoost.
  - Create an automated pipeline for data ingestion and churn prediction.
