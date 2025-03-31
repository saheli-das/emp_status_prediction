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
    best_model_app = joblib.load('best_model_app.joblib')
    transformer_app = joblib.load('transformer_app.joblib')
    important_features_app = joblib.load('important_features_app.joblib')
    percentiles_app = joblib.load('percentiles_app.joblib')
    dept_list = joblib.load('dept_list.joblib')  # Load department names from training data

    st.title("Prediction App")
    option = st.radio("Choose input method:", ("Manual Input", "Upload CSV"))

    if option == "Manual Input":
        st.sidebar.header("Input Features")
        tenure = st.sidebar.number_input("Tenure (in months)", min_value=0, max_value=1000, value=60)
        age = st.sidebar.number_input("Age (>=20)", min_value=20, max_value=100, value=40)
        sex = st.sidebar.selectbox("Sex", ["M", "F"])
        no_of_projects = st.sidebar.number_input("Number of Projects", min_value=1, max_value=100, value=5)
        salary = st.sidebar.number_input("Salary", min_value=0, value=50000)
        last_performance_rating = st.sidebar.selectbox("Last Performance Rating", ['PIP', 'C', 'B', 'S', 'A'])
        title = st.sidebar.selectbox("Title", [
            "Manager", "Engineer", "Technique Leader", "Staff", 
            "Senior Engineer", "Senior Staff", "Assistant Engineer"
        ])
        dept_names = st.sidebar.multiselect("Department Names", dept_list)
        no_of_departments = len(dept_names)

        input_data = {
            "tenure": tenure,
            "age": age,
            "sex": sex,
            "no_of_projects": no_of_projects,
            "salary": salary,
            "Last_performance_rating": last_performance_rating,
            "title": title,
            "no_of_departments": no_of_departments
        }
        input_df = pd.DataFrame([input_data])

        # Tenure categorization
        adjusted_bins = percentiles_app.copy()
        adjusted_bins[0] = -float('inf')
        adjusted_bins[-1] = float('inf')
        input_df['tenure_category'] = pd.cut(input_df['tenure'], bins=adjusted_bins,
                                             labels=['Low', 'Medium', 'High', 'Very High'], include_lowest=True)
        bins = [20, 35, 50, float('inf')]
        labels = ['20-35', '35-50', '50+']
        input_df['age_group'] = pd.cut(input_df['age'], bins=bins, labels=labels, right=False)

        # One-hot encoding for departments
        for dept in dept_list:
            input_df[f'dept_{dept}'] = 1 if dept in dept_names else 0

        if st.button("Predict"):
            input_transformed = transformer_app.transform(input_df)
            input_transformed_df = pd.DataFrame(input_transformed, columns=transformer_app.get_feature_names_out())
            input_filtered = input_transformed_df[important_features_app]
            predictions = best_model_app.predict(input_filtered)
            prediction_label = "leave" if predictions[0] == 1 else "stay"
            st.header("Prediction Result")
            st.write(f"The predicted output is: **{prediction_label}**")

    elif option == "Upload CSV":
        st.header("Upload CSV File")
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

        if uploaded_file is not None:
            csv_df = pd.read_csv(uploaded_file)
            required_columns = [
                "hire_date", "last_date", "birth_date", "sex", "no_of_projects", 
                "salary", "Last_performance_rating", "title", "dept_names", "no_of_departments", "tenure", "age"
            ]
            if not all(column in csv_df.columns for column in required_columns):
                st.error(f"The CSV file must contain the following columns: {required_columns}")
            else:
                st.write("Processing CSV file...")
                csv_df.drop(columns=['emp_no', 'first_name', 'last_name', 'emp_title_id', 'dept_nos', 'birth_date', 'last_date', 'hire_date'], inplace=True, errors='ignore')
                csv_df['tenure_category'] = pd.cut(csv_df['tenure'], bins=percentiles_app, labels=['Low', 'Medium', 'High', 'Very High'], include_lowest=True)
                csv_df['age_group'] = pd.cut(csv_df['age'], bins=[20, 35, 50, float('inf')], labels=['20-35', '35-50', '50+'], right=False)
                
                # One-hot encoding for department names
                for dept in dept_list:
                    csv_df[f'dept_{dept}'] = csv_df['dept_names'].apply(lambda x: 1 if dept in str(x) else 0)
                
                csv_df.drop(columns=['tenure', 'age', 'dept_names'], inplace=True)
                input_transformed = transformer_app.transform(csv_df)
                input_transformed_df = pd.DataFrame(input_transformed, columns=transformer_app.get_feature_names_out())
                input_filtered = input_transformed_df[important_features_app]
                predictions = best_model_app.predict(input_filtered)
                probabilities = best_model_app.predict_proba(input_filtered)[:, 1]
                csv_df['prediction'] = ["leave" if pred == 1 else "stay" for pred in predictions]
                csv_df['probability_of_leaving'] = probabilities
                st.dataframe(csv_df)

                result_df = csv_df.groupby(['prediction', 'sex']).size().reset_index(name='count')
                st.header("Prediction Results")
                st.write(result_df)

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=result_df, x='prediction', y='count', hue='sex', ax=ax)
                ax.set_title("Number of Employees Who Leave/Stay by Sex")
                st.pyplot(fig)

                overall_counts = result_df.groupby('prediction')['count'].sum()
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.pie(overall_counts, labels=overall_counts.index, autopct='%1.1f%%', startangle=90)
                st.pyplot(fig)

