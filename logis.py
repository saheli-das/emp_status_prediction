import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model and preprocessing objects
best_model_new = joblib.load('best_model_new.joblib')
transformer_new = joblib.load('transformer_new.joblib')
important_features_new = joblib.load('important_features_new.joblib')
percentiles_new = joblib.load('percentiles_new.joblib')

# Streamlit app title
st.title("Prediction App")

# Option to choose between manual input and CSV upload
option = st.radio("Choose input method:", ("Manual Input", "Upload CSV"))

if option == "Manual Input":
    # Input fields for user data
    st.sidebar.header("Input Features")
    tenure = st.sidebar.number_input("Tenure (in months)", min_value=0, max_value=1000, value=60)
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=40)
    sex = st.sidebar.selectbox("Sex", ["M", "F"])
    no_of_projects = st.sidebar.number_input("Number of Projects", min_value=1, max_value=100, value=5)
    salary = st.sidebar.number_input("Salary", min_value=0, value=50000)
    last_performance_rating = st.sidebar.selectbox("Last Performance Rating", ['PIP', 'C', 'B', 'S', 'A'])

    # Add missing columns with default values
    title = st.sidebar.selectbox("Title", [
        "Manager", "Engineer", "Technique Leader", "Staff", 
        "Senior Engineer", "Senior Staff", "Assistant Engineer"
    ])
    
    # Use multiselect for department names
    dept_names = st.sidebar.multiselect("Department Names", [
        "Marketing", "Human Resources", "Research", "Sales", 
        "Quality Management", "Production", "development", 
        "Finance", "Customer Service"
    ], default=["Marketing"])  # Set a default value if needed
    
    no_of_departments = st.sidebar.number_input("Number of Departments", min_value=1, max_value=10, value=2)

    # Create a dictionary from the input data
    input_data = {
        "tenure": tenure,
        "age": age,
        "sex": sex,
        "no_of_projects": no_of_projects,
        "salary": salary,
        "Last_performance_rating": last_performance_rating,
        "title": title,  # Include missing columns
        "dept_names": ", ".join(dept_names),  # Join selected departments with a comma
        "no_of_departments": no_of_departments  
    }

    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Apply the same feature engineering steps as during training
    # Create tenure_category using percentiles
    input_df['tenure_category'] = pd.cut(input_df['tenure'], bins=percentiles_new, labels=['Low', 'Medium', 'High', 'Very High'], include_lowest=True)

    # Create age_group based on bins
    bins = [20, 35, 50, float('inf')]
    labels = ['20-35', '35-50', '50+']
    input_df['age_group'] = pd.cut(input_df['age'], bins=bins, labels=labels, right=False)

    # Add a predict button
    if st.button("Predict"):
        # Transform the input data using the saved transformer
        input_transformed = transformer_new.transform(input_df)

        # Create a DataFrame from the transformed data
        input_transformed_df = pd.DataFrame(input_transformed, columns=transformer_new.get_feature_names_out())

        # Filter features based on importance
        input_filtered = input_transformed_df[important_features_new]

        # Make predictions
        predictions = best_model_new.predict(input_filtered)

        # Map prediction value to label
        prediction_label = "left" if predictions[0] == 1 else "stayed"

        # Display the prediction
        st.header("Prediction Result")
        st.write(f"The predicted output is: **{prediction_label}**")
    else:
        st.write("Click the **Predict** button to see the result.")

elif option == "Upload CSV":
    st.header("Upload CSV File")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the CSV file into a DataFrame
        csv_df = pd.read_csv(uploaded_file)

        # Check if required columns are present
        required_columns = [
            "hire_date", "last_date", "birth_date", "sex", "no_of_projects", 
            "salary", "Last_performance_rating", "title", "dept_names", "no_of_departments"
        ]
        if not all(column in csv_df.columns for column in required_columns):
            st.error(f"The CSV file must contain the following columns: {required_columns}")
        else:
            # Show progress
            st.write("Processing CSV file...")
            progress_bar = st.progress(0)

            # Convert date columns to datetime
            csv_df['hire_date'] = pd.to_datetime(csv_df['hire_date'])
            csv_df['last_date'] = pd.to_datetime(csv_df['last_date'])
            csv_df['birth_date'] = pd.to_datetime(csv_df['birth_date'])

            # Calculate tenure and age
            max_last_date = csv_df['last_date'].max()
            csv_df['tenure'] = (csv_df['last_date'].fillna(max_last_date) - csv_df['hire_date']).dt.days / 365
            csv_df['age'] = (csv_df['last_date'].fillna(max_last_date) - csv_df['birth_date']).dt.days / 365

            # Apply additional steps
            csv_df['tenure'] = (csv_df['tenure'] * 12).round().astype(int)  # Convert tenure to months
            csv_df['age'] = csv_df['age'].round().astype(int)  # Round age to the nearest integer

            # Convert 'left' column to integer (if it exists)
            if 'left' in csv_df.columns:
                csv_df['left'] = csv_df['left'].astype(int)

            # Drop unnecessary columns
            columns_to_drop = ['emp_no', 'first_name', 'last_name', 'emp_title_id', 'dept_nos', 'birth_date', 'last_date', 'hire_date', 'left']
            csv_df.drop(columns=[col for col in columns_to_drop if col in csv_df.columns], inplace=True)

            # Apply the same feature engineering steps as during training
            csv_df['tenure_category'] = pd.cut(csv_df['tenure'], bins=percentiles_new, labels=['Low', 'Medium', 'High', 'Very High'], include_lowest=True)
            bins = [20, 35, 50, float('inf')]
            labels = ['20-35', '35-50', '50+']
            csv_df['age_group'] = pd.cut(csv_df['age'], bins=bins, labels=labels, right=False)

            # Display the processed data
            st.write("Processed Data:")
            st.write(csv_df)

            # Transform the entire DataFrame at once
            st.write("Transforming data...")
            input_transformed = transformer_new.transform(csv_df)
            input_transformed_df = pd.DataFrame(input_transformed, columns=transformer_new.get_feature_names_out())

            # Filter features based on importance
            input_filtered = input_transformed_df[important_features_new]

            # Make predictions for the entire DataFrame
            st.write("Making predictions...")
            predictions = best_model_new.predict(input_filtered)

            # Add predictions to the DataFrame
            csv_df['prediction'] = ["left" if pred == 1 else "stayed" for pred in predictions]

            # Update progress
            progress_bar.progress(100)

            # Aggregate results
            result_df = csv_df.groupby(['prediction', 'sex']).size().reset_index(name='count')

            # Display the aggregated results
            st.header("Prediction Results")
            st.write(result_df)

            # Visualize the results
            st.header("Visualization")

            # Bar chart: Number of employees who left/stayed by sex
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=result_df, x='prediction', y='count', hue='sex', ax=ax)
            ax.set_title("Number of Employees Who Left/Stayed by Sex")
            ax.set_xlabel("Prediction")
            ax.set_ylabel("Count")
            st.pyplot(fig)

            # Pie chart: Overall left vs stayed
            overall_counts = result_df.groupby('prediction')['count'].sum()
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(overall_counts, labels=overall_counts.index, autopct='%1.1f%%', startangle=90)
            ax.set_title("Overall Left vs Stayed")
            st.pyplot(fig)
