# Supervised-Machine-Learning-Challlenge

This repository contains my solution of the Supervised Machine Learning Homework - Predicting Credit Risk of the GATECH Data Science and Analytics Bootcamp. The goal was to build a machine learning model that attempts to predict whether a loan from LendingClub will become risk or not.

## Background

LendingClub is a peer-to-peer lending services company that allows individual investors to partially fund personal loans as well as buy and sell notes backing the loans on a secondary market. LendingClub offers their previous data through an API.
I used that data to create machine learning models to classify the risk level of given loans. Specifically, comparing the Logistic Regression model and Random Forest Classifier.

## Steps

### 1. Data Retrieval

Using the [Generator Jupyter Notebook](Resources/Generator/GenerateData.ipynb), 2 csv files were generated:
- [2019loans.csv](Resources/2019loans.csv).
- [2020Q1loans.csv](Resources/2020Q1loans.csv).

Note: these two CSVs have been undersampled to give an even number of high risk and low risk loans. In the original dataset, only 2.2% of loans are categorized as high risk. To get a truly accurate model, special techniques need to be used on imbalanced data. Undersampling is one of those techniques. Oversampling and SMOTE (Synthetic Minority Over-sampling Technique) are other techniques that are also used.

### 2. Preprocessing: Convert categorical data to numeric

Created a training set from the 2019 loans using pd.get_dummies() to convert the categorical data to numeric columns. Similarly, created a testing set from the 2020 loans, also using pd.get_dummies(). Note! There were categories in the 2019 loans that did not exist in the testing set. I needed to use code to fill in the missing categories in the testing set.

### 3. Consider the models

Created and compared two models on this data: a logistic regression, and a random forests classifier. Before, creating, fitting, and scoring the models, I made a prediction as to which model I thought will perform better. My initial guess was that the Random Forest Classifier would do better than the Logistic Regression given the number of features, it was best to combine a number of decision trees on different subsets of a dataset and average the results to increase the dataset's predicted accuracy.It would also overcome the problem of decision tree overfitting.

### 4. Fit a LogisticRegression model and RandomForestClassifier model

Created a LogisticRegression model, fitted it to the data, and printed the model's score. Done the same for a RandomForestClassifier. 

### 5. Revisit the Preprocessing: Scale the data

Refitted the previous models on a scaled data using `StandardScaler`.

## References
LendingClub (2019-2020) Loan Stats. Retrieved from: https://resources.lendingclub.com/