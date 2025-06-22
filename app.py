import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import requests

st.title('Heart Disease Prediction Dashboard')

    # Load the trained Random Forest model
model = joblib.load('random_forest_model.joblib')

    # Dataset Upload
uploaded_file = st.file_uploader("Upload your heart disease dataset", type=['csv', 'xlsx'])
if uploaded_file is not None:
        # Read the uploaded file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

        # Display Dataset
    st.write("### Dataset Preview")
    st.dataframe(df.head())

        # Convert object-type columns to numeric (int/float) if necessary
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Data Visualizations
    st.write("### Data Visualization")
    if st.button("Generate Visualizations"):
        st.write("#### Heatmap of Correlations")
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
        st.pyplot(plt)

        st.write("#### Boxplot of Features")
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df)
        st.pyplot(plt)

        st.write('Chol Distribution by Predicted Attribute (num)')
        plt.figure(figsize=(8, 6))
        sns.histplot(x='chol', hue='num', data=df)
        plt.xticks(rotation=90)
        st.pyplot(plt)

        # Sidebar Input Features for Prediction
    st.sidebar.header("Input Features")
    def user_input_features():
        age = st.sidebar.slider('Age', min_value=0, max_value=120, value=50)
        sex = st.sidebar.selectbox('Sex (1 = Male, 0 = Female)', [0, 1])
        dataset = st.sidebar.selectbox('City Name Type (0-3)', [0, 1, 2, 3])
        cp = st.sidebar.selectbox('Chest Pain Type (0-3)', [0, 1, 2, 3])
        trestbps = st.sidebar.slider('Resting Blood Pressure', min_value=80, max_value=200, value=120)
        chol = st.sidebar.slider('Serum Cholesterol in mg/dl', min_value=100, max_value=600, value=200)
        fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)', [0, 1])
        restecg = st.sidebar.selectbox('Resting ECG (0-2)', [0, 1, 2])
        thalach = st.sidebar.slider('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150)
        exang = st.sidebar.selectbox('Exercise Induced Angina (1 = Yes, 0 = No)', [0, 1])
        oldpeak = st.sidebar.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=6.0, value=1.0, step=0.1)
        slope = st.sidebar.selectbox('Slope of Peak Exercise ST Segment (0-2)', [0, 1, 2])
        ca = st.sidebar.number_input('Number of Major Vessels Colored by Fluoroscopy (0-4)', min_value=0, max_value=4, value=0)
        thal = st.sidebar.selectbox('Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)', [1, 2, 3])

            # Create a dictionary of inputs
        data = {
            'age': age, 'sex': sex, 'dataset': dataset, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 
            'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang, 
            'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
        }
        features = pd.DataFrame(data, index=[0])
        features.rename(columns={'thalach': 'thalch'}, inplace=True)
        return features

        # Get user inputs from sidebar
    input_df = user_input_features()

        # Display input features
    st.write("### User Input Features")
    st.write(input_df)

    level_labels = {
        0: "No Heart Disease",
        1: "Minor Heart Disease",
        2: "Moderate Heart Disease",
        3: "Severe Heart Disease",
        4: "Critical Heart Disease"
    }

        # Prediction and Save Button
    # Prediction and Save Button
if st.button("Predict & Save"):
    try:
        # Make predictions using the Random Forest model
        prediction = model.predict(input_df)
        level = prediction[0]

        # Display prediction result
        st.write(f'Heart Disease Level: {level} : {level_labels.get(level, "Unknown")}')

        # Convert dataframe to a dictionary
        input_data = input_df.to_dict(orient='records')[0]  # Convert to a single dictionary

        # Add the prediction result to the data
        input_data['prediction'] = int(level)

        # Send data to FastAPI for saving in database
        api_url = "http://127.0.0.1:8080/predict/"
        response = requests.post(api_url, json=input_data)  # Sending as JSON

        if response.status_code == 200:
            st.success("Data has been saved to the database successfully!")
        else:
            st.error("Failed to save data to the database!")
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
