# emp_status_prediction
This app predicts if an employee/employees will stay in the company or leave the company based on data entered manually or by uploading csv file for more than one employees
**Steps for creation of the App**
**Load Saved Artifacts:**
Load the trained model (best_model_new.joblib), preprocessing pipeline (transformer_new.joblib), important features (important_features_new.joblib), and percentiles (percentiles_new.joblib).
**Set Up Streamlit Interface:**
Add a title and radio button for input method selection: Manual Input or Upload CSV.
**Manual Input Mode:**
Create input fields for features like tenure, age, sex, no_of_projects, salary, and last_performance_rating.
Add missing columns (title, dept_names, no_of_departments) with default values.
Apply feature engineering (e.g., tenure_category, age_group) and preprocessing.
Predict using the trained model and display the result.
**CSV Upload Mode:**
Allow users to upload a CSV file.
Process the CSV: convert dates, calculate tenure and age, drop unnecessary columns, and apply feature engineering.
Transform and filter data using the preprocessing pipeline.
Make predictions for the entire dataset and display results.
**Visualize Results:**
Aggregate predictions by prediction and sex.
Display bar charts and pie charts for insights
