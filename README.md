# Supervised Machine Learning Homework - Predicting Credit Risk

In this project, I built a machine learning model that attempts to predict whether or not a lone will become high-risk. 

## Background

The purpose of this project is to evaluate a subset of a lending company's loan data that is available through an API call.  The purpose is to use a machine leanring model to classify the risk level of given laons by comparing the Logistic Regression model with the Random Forest Classifer fo both scaled and unscaled data.  

### Retrieve the data

A [GenerateData.ipynb](/Resources/Generator/GenerateData.ipynb) script has been put together in the `Generator` folder within the `Resources` folder that was designed to automate the retrieval of lending data that produces an output of two CSVs: 

* `2019loans.csv`
* `2020Q1loans.csv`

For this machine learning model, an entire year's worth of data (2019) will be used to predict the credit risk of loans from the first quarter of the following year (2020).

Because the CSVs have been undersampled to give an even number of high risk and low risk loans, undersampling was used because techniques are required when the data is imbalanced. 

## Preprocessing: 

A training dataset was created from the 2019 loans using `pd.get_dummies()`.  The categorical data was converted to numeric columns. Similarly, a testing dataset was created from the 2020 loan data, also by using `pd.get_dummies()`. Because there are categories in the 2019 loans that do not exist in the testing set, code was used to fill in the missing categories in the testing dataset.  That way, it allowed the possibility of fitting a model to the training set and try to score it on the testing set. You need to use code to fill in the missing categories in the testing set. 

    `# Convert categorical data to numeric and separate target feature for testing data
    test_df_sample = pd.get_dummies(test_df)
    test_df_x = test_df_sample.drop(['loan_status_low_risk', 'loan_status_high_risk'], axis=1)
    test_df_y = test_df_sample['loan_status_low_risk'].values
    print(test_df_x.head())
    print(f"train_df_y array is {train_df_y}.")`

## Consider the models

Two models were used for this dataset: a logistic regression, and a random forests classifier. Before the model was fit, scored, and predicted, a pre-prediction is listed in the Jupyter Notebook as well as a summary of the educated guess. 

## Fit a LogisticRegression model and RandomForestClassifier model

Create a LogisticRegression model, fit it to the data, and print the model's score. Do the same for a RandomForestClassifier. You may choose any starting hyperparameters you like. Which model performed better? How does that compare to your prediction? Write down your results and thoughts.

## Revisit the Preprocessing: Scale the data

The data going into these models was never scaled, an important step in preprocessing. Use `StandardScaler` to scale the training and testing sets. Before re-fitting the LogisticRegression and RandomForestClassifier models on the scaled data, make another prediction about how you think scaling will affect the accuracy of the models. Write your predictions down and provide justification.

Fit and score the LogisticRegression and RandomForestClassifier models on the scaled data. How do the model scores compare to each other, and to the previous results on unscaled data? How does this compare to your prediction? Write down your results and thoughts.

## Rubric

[Unit 19 - Supervised Machine Learning Homework Rubric](https://docs.google.com/document/d/1f_eN3TYiGqlaWL9Utk5U-P491OeWqFSiv7FIlI_d4_U/edit?usp=sharing)

### References

LendingClub (2019-2020) _Loan Stats_. Retrieved from: [https://resources.lendingclub.com/](https://resources.lendingclub.com/)

- - -

© 2021 Trilogy Education Services, a 2U, Inc. brand. All Rights Reserved.
