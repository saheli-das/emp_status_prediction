import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Sidebar Login
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    st.sidebar.title("🔐 Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if username == "adminSaheli" and password == "25@das20":
            st.session_state["logged_in"] = True
            st.sidebar.success("✅ Logged in successfully!")
            st.rerun()
        else:
            st.sidebar.error("❌ Invalid credentials. Try again.")

# Show App Functionality Only If Logged In
if st.session_state["logged_in"]:

    # Load the saved model and preprocessing objects
    best_model_app = joblib.load('best_model_app.joblib')
    transformer_app = joblib.load('transformer_app.joblib')
    important_features_app = joblib.load('important_features_app.joblib')
    percentiles_app = joblib.load('percentiles_app.joblib')

    # Streamlit app title
    st.title("Prediction App")

    # Option to choose between manual input and CSV upload
    option = st.radio("Choose input method:", ("Manual Input", "Upload CSV"))

    if option == "Manual Input":
        # Input fields for user data
        st.sidebar.header("Input Features")
        tenure = st.sidebar.number_input("Tenure (in months)", min_value=0, max_value=1000, value=60)
        age = st.sidebar.number_input("Age(>=20)", min_value=20, max_value=100, value=40)
        sex = st.sidebar.selectbox("Sex", ["M", "F"])
        no_of_projects = st.sidebar.number_input("Number of Projects", min_value=1, max_value=100, value=5)
        salary = st.sidebar.number_input("Salary", min_value=0, value=50000)
        last_performance_rating = st.sidebar.selectbox("Last Performance Rating", ['PIP', 'C', 'B', 'S', 'A'])

        # Add missing columns with default values
        title = st.sidebar.selectbox("Title", [
            "Manager", "Engineer", "Technique Leader", "Staff", 
            "Senior Engineer", "Senior Staff", "Assistant Engineer"
        ])
        
        dept_names = st.sidebar.selectbox("Department Name", [
            "Marketing", "Human Resources", "Research", "Sales", 
            "Quality Management", "Production", "development", 
            "Finance", "Customer Service"
        ], index=0)
        
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
            "dept_names": dept_names),  # Join selected departments with a comma
            "no_of_departments": no_of_departments  # Include missing columns
        }

        # Convert input data to a DataFrame
        input_df = pd.DataFrame([input_data])

        # Apply the same feature engineering steps as during training
       
       
        
       
        # 2. Create adjusted bins that handle all cases:
        adjusted_bins = percentiles_app.copy()
        adjusted_bins[0] = -float('inf')  # Catch any values below original minimum
        adjusted_bins[-1] = float('inf')   # Catch any values above original maximum
        
        # 3. Apply the categorization
        input_df['tenure_category'] = pd.cut(
            input_df['tenure'],
            bins=adjusted_bins,
            labels=['Low', 'Medium', 'High', 'Very High'],
            include_lowest=True
        )
        
        # 4. Handle any remaining NaN values (just in case)
        input_df['tenure_category'] = input_df['tenure_category'].cat.add_categories(['Invalid']).fillna('Invalid')

       
        
       
        
       
       # Create age_group based on bins
        bins = [20, 35, 50, float('inf')]
        labels = ['20-35', '35-50', '50+']
        input_df['age_group'] = pd.cut(input_df['age'], bins=bins, labels=labels, right=False)

        # Add a predict button
        if st.button("Predict"):
            # Transform the input data using the saved transformer
            input_transformed = transformer_app.transform(input_df)

            # Create a DataFrame from the transformed data
            input_transformed_df = pd.DataFrame(input_transformed, columns=transformer_app.get_feature_names_out())

            # Filter features based on importance
            input_filtered = input_transformed_df[important_features_app]

            # Make predictions
            predictions = best_model_app.predict(input_filtered)

            # Map prediction value to label
            prediction_label = "leave" if predictions[0] == 1 else "stay"

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
                "salary", "Last_performance_rating", "title", "dept_names", "no_of_departments", "tenure", "age"
            ]
            if not all(column in csv_df.columns for column in required_columns):
                st.error(f"The CSV file must contain the following columns: {required_columns}")
            else:
                # Show progress
                st.write("Processing CSV file...")
                progress_bar = st.progress(0)

                # Drop unnecessary columns (including 'left' if it exists)
                columns_to_drop = ['emp_no', 'first_name', 'last_name', 'emp_title_id', 'dept_nos', 'birth_date', 'last_date', 'hire_date']
                columns_to_drop = [col for col in columns_to_drop if col in csv_df.columns]
                csv_df.drop(columns=columns_to_drop, inplace=True)

                # Apply the same feature engineering steps as during training
                csv_df['tenure_category'] = pd.cut(csv_df['tenure'], bins=percentiles_app, labels=['Low', 'Medium', 'High', 'Very High'], include_lowest=True)
                bins = [20, 35, 50, float('inf')]
                labels = ['20-35', '35-50', '50+']
                csv_df['age_group'] = pd.cut(csv_df['age'], bins=bins, labels=labels, right=False)



                # Step 3: Drop Unnecessary Columns
                columns_to_remove = ['tenure', 'age']
                csv_df.drop(columns=columns_to_remove, inplace=True)

                # Transform the entire DataFrame at once
                st.write("Transforming data...")
                input_transformed = transformer_app.transform(csv_df)
                input_transformed_df = pd.DataFrame(input_transformed, columns=transformer_app.get_feature_names_out())



                # Filter features based on importance
                input_filtered = input_transformed_df[important_features_app]



                # Make predictions for the entire DataFrame
                st.write("Making predictions...")
                predictions = best_model_app.predict(input_filtered)

                # Get predicted probabilities for the "left" class (class 1)
                probabilities = best_model_app.predict_proba(input_filtered)[:, 1]

                # Add predictions and probabilities to the DataFrame
                csv_df['prediction'] = ["leave" if pred == 1 else "stay" for pred in predictions]
                csv_df['probability_of_leaving'] = probabilities  # Add probability of leaving

                # Display DataFrame in Streamlit
                st.write("Predictions with Probabilities:")
                st.dataframe(csv_df)

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
                ax.set_title("Number of Employees Who Leave/Stay by Sex")
                ax.set_xlabel("Prediction")
                ax.set_ylabel("Count")
                st.pyplot(fig)

                # Pie chart: Overall left vs stayed
                overall_counts = result_df.groupby('prediction')['count'].sum()
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.pie(overall_counts, labels=overall_counts.index, autopct='%1.1f%%', startangle=90)
                ax.set_title("Overall Leave vs Stay")
                st.pyplot(fig)
