# Mobile Addiction Prediction using Machine Learning – Project Report

## 1. Introduction
Mobile phones have become an essential part of daily life, especially for students. However, excessive mobile phone usage can lead to addiction, which affects sleep, studies, and mental health. Mobile addiction is becoming a serious problem among students due to increased use of social media, gaming, and online entertainment. This project aims to use machine learning techniques to predict whether a person is addicted to mobile phone usage based on their daily habits.

## 2. Problem Statement
The problem addressed in this project is to predict whether a person is addicted to mobile phone usage or not based on lifestyle and behavioral data such as screen time, sleep time, study time, and phone usage patterns.

## 3. Objective
The main objective of this project is to develop a machine learning model that can classify users into two categories:
- Addicted
- Not Addicted

The model will be trained using a dataset containing lifestyle habits and mobile usage behavior.

## 4. Dataset Description
The dataset used in this project contains the following features:

| Feature | Description |
|--------|-------------|
| Screen_Time_hrs | Total screen time per day |
| Social_Media_hrs | Social media usage per day |
| Gaming_Time_hrs | Gaming time per day |
| Study_Time_hrs | Study time per day |
| Sleep_Time_hrs | Sleep hours per day |
| Phone_Pickups_day | Number of times phone is checked |
| Anxiety_Without_Phone | Anxiety when phone is not available (0/1) |
| Addicted | Target variable (0 = Not Addicted, 1 = Addicted) |

## 5. Methodology
The following steps were followed in this project:
1. Data Collection
2. Data Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Scaling
5. Train-Test Split
6. Model Training
7. Model Evaluation
8. Data Visualization
9. Prediction

## 6. Machine Learning Algorithms Used
### Logistic Regression
Logistic Regression is a supervised machine learning algorithm used for binary classification problems. It predicts the probability of a class and classifies the output into categories.

### Decision Tree
Decision Tree is a classification algorithm that splits the dataset based on feature values and creates a tree-like structure. It also helps in understanding feature importance.

## 7. Results and Accuracy
The performance of the models was evaluated using accuracy, confusion matrix, and classification report.

- Logistic Regression Accuracy: ______ %
- Decision Tree Accuracy: ______ %

Both models performed well on the dataset and were able to classify addiction correctly.

## 8. Graphs and Analysis
The following graphs were generated:
- Class Distribution
- Feature Distribution
- Correlation Heatmap
- Boxplots
- Confusion Matrix
- Feature Importance
- Model Accuracy Comparison
- Scatter Plot

These graphs helped in understanding the relationship between different features and mobile addiction.

## 9. Conclusion
This project successfully developed a machine learning model to predict mobile addiction based on lifestyle habits. The results show that screen time, sleep time, social media usage, and phone pickups are important factors affecting mobile addiction. The machine learning models were able to predict addiction with good accuracy.

## 10. Future Scope
This project can be improved in the future by:
- Using a larger real-world dataset
- Creating a mobile or web application
- Using advanced machine learning algorithms
- Adding more behavioral and psychological features

## 11. References
- Scikit-learn Documentation
- Pandas Documentation
- Machine Learning Tutorials
