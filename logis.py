import streamlit as st
import joblib
import pandas as pd
import numpy as np
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
    try:
        best_model_app = joblib.load('best_model_app.joblib')
        transformer_app = joblib.load('transformer_app.joblib')
        important_features_app = joblib.load('important_features_app.joblib')
        percentiles_app = joblib.load('percentiles_app.joblib')
        
        # Get all department names the model was trained on
        transformer_features = transformer_app.get_feature_names_out()
        trained_departments = [f.replace('dept_names_', '') 
                             for f in transformer_features if f.startswith('dept_names_')]
        
        # If no departments found in model, use a default list
        if not trained_departments:
            st.warning("No department features found in model. Using default departments.")
            trained_departments = ["Marketing", "HR", "Sales", "Finance"]  # Default fallback
        
    except Exception as e:
        st.error(f"Failed to load model files: {str(e)}")
        st.stop()

    # Streamlit app title
    st.title("Employee Status Prediction App")

    # Option to choose between manual input and CSV upload
    option = st.radio("Choose input method:", ("Manual Input", "Upload CSV"))

    if option == "Manual Input":
        # Input fields for user data
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
        
        # Department selection with safe default
        default_dept = trained_departments[0] if trained_departments else ""
        dept_names = st.sidebar.multiselect(
            "Department Names", 
            trained_departments, 
            default=[default_dept] if default_dept else []
        )
        no_of_departments = st.sidebar.number_input("Number of Departments", min_value=1, max_value=10, value=len(dept_names) if dept_names else 1

        # Create a dictionary from the input data
        input_data = {
            "tenure": tenure,
            "age": age,
            "sex": sex,
            "no_of_projects": no_of_projects,
            "salary": salary,
            "Last_performance_rating": last_performance_rating,
            "title": title,
            "dept_names": ", ".join(dept_names) if dept_names else default_dept,
            "no_of_departments": no_of_departments
        }

        # Convert input data to a DataFrame
        input_df = pd.DataFrame([input_data])

        # Apply feature engineering
        adjusted_bins = percentiles_app.copy()
        adjusted_bins[0] = -float('inf')
        adjusted_bins[-1] = float('inf')
        
        input_df['tenure_category'] = pd.cut(
            input_df['tenure'],
            bins=adjusted_bins,
            labels=['Low', 'Medium', 'High', 'Very High'],
            include_lowest=True
        )
        input_df['tenure_category'] = input_df['tenure_category'].cat.add_categories(['Invalid']).fillna('Invalid')

        bins = [20, 35, 50, float('inf')]
        labels = ['20-35', '35-50', '50+']
        input_df['age_group'] = pd.cut(input_df['age'], bins=bins, labels=labels, right=False)

        # Add a predict button
        if st.button("Predict"):
            try:
                # Make a copy of the input data
                input_df_transformed = input_df.copy()
                
                # Create all expected department columns
                for dept in trained_departments:
                    input_df_transformed[f"dept_names_{dept}"] = input_df_transformed['dept_names'].apply(
                        lambda x: 1 if dept in str(x).split(', ') else 0
                    )
                
                # Now safely drop the original columns
                cols_to_drop = ['dept_names', 'tenure', 'age']
                input_df_transformed.drop(columns=[col for col in cols_to_drop if col in input_df_transformed.columns], inplace=True)
                
                # Transform the input data
                input_transformed = transformer_app.transform(input_df_transformed)
                input_transformed_df = pd.DataFrame(input_transformed, columns=transformer_app.get_feature_names_out())

                # Filter features based on importance
                input_filtered = input_transformed_df[important_features_app]

                # Make predictions
                predictions = best_model_app.predict(input_filtered)
                probabilities = best_model_app.predict_proba(input_filtered)

                # Display results
                prediction_label = "leave" if predictions[0] == 1 else "stay"
                probability = probabilities[0][1] if predictions[0] == 1 else probabilities[0][0]
                
                st.header("Prediction Result")
                st.write(f"The predicted output is: **{prediction_label}**")
                st.write(f"Probability: {probability:.2%}")
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
        else:
            st.write("Click the **Predict** button to see the result.")

    elif option == "Upload CSV":
        st.header("Upload CSV File")
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

        if uploaded_file is not None:
            # Read the CSV file
            csv_df = pd.read_csv(uploaded_file)

            # Check required columns
            required_columns = [
                "hire_date", "last_date", "birth_date", "sex", "no_of_projects", 
                "salary", "Last_performance_rating", "title", "dept_names", "no_of_departments", "tenure", "age"
            ]
            if not all(col in csv_df.columns for col in required_columns):
                st.error(f"Missing required columns: {set(required_columns) - set(csv_df.columns)}")
            else:
                # Process data
                with st.spinner('Processing data...'):
                    # Drop unnecessary columns (but keep dept_names)
                    cols_to_drop = ['emp_no', 'first_name', 'last_name', 'emp_title_id', 
                                  'dept_nos', 'birth_date', 'last_date', 'hire_date']
                    csv_df.drop(columns=[col for col in cols_to_drop if col in csv_df.columns], inplace=True)

                    # Feature engineering
                    csv_df['tenure_category'] = pd.cut(
                        csv_df['tenure'],
                        bins=percentiles_app,
                        labels=['Low', 'Medium', 'High', 'Very High'],
                        include_lowest=True
                    )
                    csv_df['age_group'] = pd.cut(
                        csv_df['age'],
                        bins=[20, 35, 50, float('inf')],
                        labels=['20-35', '35-50', '50+'],
                        right=False
                    )
                    
                    # Create department columns
                    for dept in trained_departments:
                        csv_df[f"dept_names_{dept}"] = csv_df['dept_names'].apply(
                            lambda x: 1 if str(dept) in str(x).split(', ') else 0
                        )
                    
                    # Now drop processed columns
                    csv_df.drop(columns=['dept_names', 'tenure', 'age'], inplace=True)

                    # Transform data
                    input_transformed = transformer_app.transform(csv_df)
                    input_transformed_df = pd.DataFrame(
                        input_transformed, 
                        columns=transformer_app.get_feature_names_out()
                    )

                    # Make predictions
                    predictions = best_model_app.predict(input_transformed_df[important_features_app])
                    probabilities = best_model_app.predict_proba(input_transformed_df[important_features_app])[:, 1]

                    # Add results to DataFrame
                    csv_df['prediction'] = ['leave' if p == 1 else 'stay' for p in predictions]
                    csv_df['probability'] = probabilities

                    # Show results
                    st.success("Processing complete!")
                    st.dataframe(csv_df)

                    # Visualizations
                    st.header("Analysis")
                    
                    # Prediction distribution
                    fig1, ax1 = plt.subplots()
                    csv_df['prediction'].value_counts().plot(kind='bar', ax=ax1)
                    ax1.set_title("Prediction Distribution")
                    st.pyplot(fig1)
                    
                    # Probability distribution
                    fig2, ax2 = plt.subplots()
                    sns.histplot(csv_df['probability'], bins=20, ax=ax2)
                    ax2.set_title("Probability Distribution")
                    st.pyplot(fig2)
