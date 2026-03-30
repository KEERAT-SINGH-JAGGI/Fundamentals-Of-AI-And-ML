# Mobile Addiction Prediction using Machine Learning

## Project Overview
This project predicts whether a person is addicted to mobile phone usage or not based on daily lifestyle habits such as screen time, social media usage, gaming time, study time, sleep time, phone pickups per day, and anxiety without phone. The project uses machine learning classification algorithms to analyze the data and make predictions.

## Problem Statement
Mobile phone addiction is increasing among students and affects sleep, studies, and mental health. The goal of this project is to predict mobile addiction using machine learning based on lifestyle habits.

## Objective
The objective of this project is to build a machine learning model that can classify whether a person is addicted to mobile phone usage or not addicted based on behavioral and lifestyle data.

## Machine Learning Algorithms Used
- Logistic Regression
- Decision Tree Classifier

## Technologies Used
- Python
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Scikit-learn

## Dataset Features
The dataset contains the following features:
- Screen_Time_hrs
- Social_Media_hrs
- Gaming_Time_hrs
- Study_Time_hrs
- Sleep_Time_hrs
- Phone_Pickups_day
- Anxiety_Without_Phone
- Addicted (Target Variable)

## How to Run the Project
1. Install required libraries:
   pip install -r requirements.txt

2. Run the Python file:
   python mobileaddictionpredictor.py

3. The program will:
   - Load the dataset
   - Train machine learning models
   - Calculate accuracy
   - Generate graphs
   - Predict addiction for new users
   - Save graphs in the outputs folder

## Output
The program outputs:
- Accuracy of Logistic Regression and Decision Tree models
- Confusion Matrix
- Classification Report
- Feature Importance Graph
- Correlation Heatmap
- Addiction Prediction for new users

## Conclusion
This project shows how machine learning can be used to analyze lifestyle habits and predict mobile addiction. Screen time, sleep time, and phone pickups were found to be important factors affecting mobile addiction.

## Future Improvements
- Use a larger real-world dataset
- Create a web or mobile app interface
- Use advanced machine learning models
- Add more features like stress level and academic performance
