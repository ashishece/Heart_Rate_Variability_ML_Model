#!/usr/bin/env python
# coding: utf-8

# In[1]:


cd Desktop


# In[2]:


cd JUNE_BIOARO_NEW_MODEL


# In[3]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# In[4]:


# Load the dataset
df = pd.read_csv('heart_rate_variability.csv')

# Ensure all necessary columns are present
required_columns = [
    'HR', 'HRV', 'Age', 'SleepDuration', 'SleepQuality', 'ActivityLevel', 
    'CaloriesBurned', 'StressLevel', 'Mood', 'SystolicBP', 'DiastolicBP', 'SPO2'
]

missing_columns = set(required_columns) - set(df.columns)
if missing_columns:
    raise ValueError(f"Missing columns in the dataset: {missing_columns}")

# Drop any potential missing values
df.dropna(inplace=True)


# In[5]:


# Feature and target variables
X = df[['HR', 'Age', 'SleepDuration', 'SleepQuality', 'ActivityLevel', 
        'CaloriesBurned', 'StressLevel', 'Mood', 'SystolicBP', 'DiastolicBP', 'SPO2']]
y = df['HRV'].apply(lambda x: 1 if x > 60 else (0 if x >= 50 else -1))  # 1: Above, 0: Normal, -1: Below

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the RandomForestClassifier model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)


# In[6]:


# Function to calculate RMSSD
def calculate_rmssd(heart_rates):
    diff = np.diff(heart_rates)
    squared_diff = np.square(diff)
    mean_squared_diff = np.mean(squared_diff)
    rmssd = np.sqrt(mean_squared_diff)
    return rmssd


# In[7]:


# Function to provide HRV prediction and recommendation
def predict_hrv(user_data):
    # Calculate RMSSD
    rmssd_hrv = calculate_rmssd(user_data['HeartRate30Days'])
    
    # Prepare input features for prediction
    user_data_mapped = {
        'HR': np.mean(user_data['HeartRate30Days']),  # Assuming 'HR' corresponds to average heart rate
        'Age': user_data['Age'],
        'SleepDuration': user_data['SleepDuration'],
        'SleepQuality': user_data['SleepQuality'],
        'ActivityLevel': user_data['ActivityLevel'],
        'CaloriesBurned': user_data.get('CaloriesBurned', 2500),  # Use default if not provided
        'StressLevel': user_data['StressLevel'],
        'Mood': user_data['Mood'],
        'SystolicBP': user_data['SystolicBP'],
        'DiastolicBP': user_data['DiastolicBP'],
        'SPO2': 98  # Assuming 'SPO2' is a constant value for now
    }
    
    user_df = pd.DataFrame([user_data_mapped])
    user_df_scaled = scaler.transform(user_df)
    
    # Predict HRV range
    predicted_hrv_category = model.predict(user_df_scaled)[0]
    if predicted_hrv_category == 1:
        recommendation = "Your HRV is above the normal range. Keep up the good work!"
    elif predicted_hrv_category == 0:
        recommendation = "Your HRV is within the normal range. Maintain your healthy habits."
    else:
        recommendation = "Your HRV is below the normal range. Consider improving your lifestyle and consult with a healthcare professional."
    
    return rmssd_hrv, recommendation


# In[8]:


def predict_disease_risks(user_data):
    # Initialize HRV if not provided in user_data
    if 'HRV' not in user_data or user_data['HRV'] is None:
        user_data['HRV'] = calculate_rmssd(user_data['HeartRate30Days'])  # Example calculation, replace as needed
    
    user_data_mapped = {
        'HR': np.mean(user_data['HeartRate30Days']),
        'Age': user_data['Age'],
        'SleepDuration': user_data['SleepDuration'],
        'SleepQuality': user_data['SleepQuality'],
        'ActivityLevel': user_data['ActivityLevel'],
        'CaloriesBurned': user_data.get('CaloriesBurned', 2500),  # Default value if not provided
        'StressLevel': user_data['StressLevel'],
        'Mood': user_data['Mood'],
        'SystolicBP': user_data['SystolicBP'],
        'DiastolicBP': user_data['DiastolicBP'],
        'SPO2': user_data['SPO2']  # Assuming 'SPO2' is a constant value for now
    }
    
    user_df = pd.DataFrame([user_data_mapped])
    user_df_scaled = scaler.transform(user_df)
    
    # For demonstration, assume some risk prediction logic
    risk_messages = []
    recommendations = []

    # Blood pressure risk
    if user_data['SystolicBP'] > 140 or user_data['DiastolicBP'] > 90:
        bp_risk = "High blood pressure"
        bp_risk_percentage = 0.8  # Example risk percentage based on user input
        risk_messages.append(('hypertension', bp_risk_percentage))
        recommendations.append(
            ("High blood pressure", 
             "To reduce blood pressure risk:\n"
             "- Reduce salt intake\n"
             "- Increase physical activity\n"
             "- Maintain a healthy weight\n"
             "- Limit alcohol consumption\n"
             "- Eat a balanced diet rich in fruits and vegetables")
        )
    else:
        bp_risk = "Normal blood pressure"
        bp_risk_percentage = 0.2  # Example low risk percentage
        risk_messages.append(('hypertension', bp_risk_percentage))

    # HRV risk assessment
    hrv_risk_percentage = (100 - user_data['HRV']) / 100.0  # Example inverse relationship
    if hrv_risk_percentage > 0.5:
        hrv_risk = "High risk of cardiovascular disease"
        risk_messages.append(('cardiovascular disease', hrv_risk_percentage))
        recommendations.append(
            ("High risk of cardiovascular disease",
             "To improve cardiovascular health:\n"
             "- Engage in regular physical activity\n"
             "- Manage stress through relaxation techniques\n"
             "- Ensure adequate and quality sleep\n"
             "- Maintain a balanced diet low in trans fats\n"
             "- Avoid smoking and limit alcohol intake")
        )
    else:
        hrv_risk = "Normal cardiovascular health"
        risk_messages.append(('cardiovascular disease', 1 - hrv_risk_percentage))

    # Diabetes risk assessment (hypothetical example based on other features)
    diabetes_risk_percentage = user_data['SleepDuration'] / 24.0  # Example relationship with sleep duration
    if diabetes_risk_percentage > 0.5:
        diabetes_risk = "Higher risk of diabetes"
        risk_messages.append(('diabetes', diabetes_risk_percentage))
        recommendations.append(
            ("Higher risk of diabetes",
             "To lower diabetes risk:\n"
             "- Maintain a healthy weight\n"
             "- Follow a balanced diet low in refined sugars\n"
             "- Engage in regular physical activity\n"
             "- Monitor blood glucose levels\n"
             "- Get regular health check-ups")
        )
    else:
        diabetes_risk = "Normal diabetes risk"
        risk_messages.append(('diabetes', 1 - diabetes_risk_percentage))

    # Depression risk assessment (hypothetical example based on mood and stress level)
    depression_risk_percentage = user_data['Mood'] / 10.0 + (10 - user_data['StressLevel']) / 10.0  # Example combined metric
    if depression_risk_percentage > 0.5:
        depression_risk = "Higher risk of depression"
        risk_messages.append(('depression', depression_risk_percentage))
        recommendations.append(
            ("Higher risk of depression",
             "To reduce depression risk:\n"
             "- Seek support from friends, family, or a healthcare professional\n"
             "- Consider therapy or counseling\n"
             "- Engage in regular physical activity\n"
             "- Practice mindfulness or relaxation techniques\n"
             "- Maintain a regular sleep schedule")
        )
    else:
        depression_risk = "Normal depression risk"
        risk_messages.append(('depression', 1 - depression_risk_percentage))

    # Normalize risk percentages to sum up to 100%
    total_risk = sum(risk[1] for risk in risk_messages)
    for i in range(len(risk_messages)):
        risk_messages[i] = (risk_messages[i][0], risk_messages[i][1] / total_risk)

    return bp_risk, hrv_risk, diabetes_risk, depression_risk, risk_messages, recommendations


# In[ ]:


# Main program
if __name__ == "__main__":
    print("Please enter your health data for the last 30 days.")
    
    user_data = {
        'Age': int(input("Age: ")),
        'HeartRate30Days': [],
        'SystolicBP': float(input("Systolic Blood Pressure (mmHg): ")),
        'DiastolicBP': float(input("Diastolic Blood Pressure (mmHg): ")),
        'SleepDuration': float(input("Average Sleep Duration (hours): ")),
        'SleepQuality': int(input("Sleep Quality (scale 1-10): ")),
        'ActivityLevel': int(input("Average Activity Level (steps): ")),
        'StressLevel': int(input("Average Stress Level (scale 1-10): ")),
        'Mood': int(input("Average Mood (scale 1-10): ")),
        'SPO2': int(input("SPO2: "))
    }
    
    # Optional input for CaloriesBurned
    calories_burned_input = input("Average Calories Burned (calories, optional): ")
    if calories_burned_input:
        user_data['CaloriesBurned'] = int(calories_burned_input)
    
    for i in range(1, 31):
        while True:
            try:
                hr = float(input(f"Heart Rate Day {i}: "))
                user_data['HeartRate30Days'].append(hr)
                break
            except ValueError:
                print("Please enter a valid number.")
    
    rmssd_hrv, hrv_recommendation = predict_hrv(user_data)
    print(f"Calculated HRV (RMSSD): {rmssd_hrv:.2f}")
    print(f"HRV Recommendation: {hrv_recommendation}")
    
    bp_risk, hrv_risk, diabetes_risk, depression_risk, risk_messages, recommendations = predict_disease_risks(user_data)
    print(f"Blood Pressure Status: {bp_risk}")
    print(f"Cardiovascular Health: {hrv_risk}")
    print(f"Diabetes Risk: {diabetes_risk}")
    print(f"Depression Risk: {depression_risk}")
    
    # Print risk messages with probabilities
    print("\nIndividual Risk Messages:")
    for category, risk_status in risk_messages:
        print(f"{category.capitalize()} risk: {risk_status*100:.2f}%")
    
    # Print recommendations
    if recommendations:
        print("\nRecommendations for Improving Health:")
        for category, recommendation in recommendations:
            print(f"\n{category}:\n{recommendation}")


# In[ ]:




