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

    test_df_sample = pd.get_dummies(test_df)
    test_df_x = test_df_sample.drop(["loan_status_low_risk", 'loan_status_high_risk'], axis=1)
    test_df_y = test_df_sample['loan_status_low_risk'].values
    print(test_df_x.head())
    print(f"train_df_y array is {train_df_y}.")


## Consider the models

Two models were used for this dataset: a logistic regression, and a random forests classifier. Before the model was fit, scored, and predicted, a pre-prediction is listed in the Jupyter Notebook as well as a summary of the educated guess. 


## Fit a LogisticRegression model and RandomForestClassifier model

After the preprocessing, a `LogisticRegression` model was used to fit the data and print the model's score for the unscaled data.  

    classifier = LogisticRegression()
    classifier
    classifier.fit(train_df_x, train_df_y)
    classifier.fit(test_df_x, test_df_y)
    print(f"The Logistic Regression model Score for the unscaled training data is  {classifier.score(train_df_x, train_df_y)}".)
    print(f"The Logistic Regression model Score for the unscaled testing data is {classifier.score(test_df_x, test_df_y)}".)


Result:  

    The Logistic Regression model Score for the unscaled training data is  0.648440065681445
    The Logistic Regression model Score for the unscaled testing data is  0.5253083794130158


Following this, a RandomForestClassifier was used to fit the data and print the model's score for the unscaled data.


    rlf_train = RandomForestClassifier(random_state=1, n_estimators=500).fit(train_df_x, train_df_y)
    print(f"The Random Forest Classifier model Score for the unscaled training data is {rlf_train.score(train_df_x, train_df_y)}.")
    print(f"The Random Forest Classifier model Score for the unscaled testing data is {rlf_train.score(test_df_x, test_df_y)}.")


Result:

    Random Forest Classifier model score for the scaled training data is 1.0
    Random Forest Classifier model score for the scaled testing data is 0.6193109315185028


The results varied considerably, so the preprocessing was revisited.

## Revisit the Preprocessing: Scale the data

The data going into these models was never scaled, which is an important step in preprocessing. The `StandardScaler` was used to scale the training and testing sets. 

    scaler = StandardScaler().fit(train_df_x)
    X_train_scaled = scaler.transform(train_df_x)
    X_test_scaled = scaler.transform(test_df_x)
    reg_train = LinearRegression().fit(X_train_scaled, train_df_y)
    print(f"Linear Regression Score for the scaled training data is {reg_train.score(X_train_scaled, train_df_y)}.")
    print(f"Linear Regression Score for the scaled testing data is {reg_train.score(X_test_scaled, test_df_y)}.")


Result:

    Linear Regression Score for the scaled training data is 0.15694581802378016.
    Linear Regression Score for the scaled testing data is -5.139575915335923e+27.


Before re-fitting the LogisticRegression and RandomForestClassifier models on the scaled data, another prediction is included prior to seeing how the scaling will affect the accuracy of the models.  

Once the data was scaled, the `LogisticRegression` and `RandomForestClassifier` models were reapplied on the scaled data. The difference was significant and seemed to imply that the data is insufficient to make a conclusion on high-risk vs. low-risk.

    clf_train_scaled = LogisticRegression().fit(X_train_scaled, train_df_y)
    print(f'The Logistic Regression Score for the scaled data is {clf_train_scaled.score(X_train_scaled, train_df_y)}')
    print(f'The Logistic Regression Score for the scaled data is {clf_train_scaled.score(X_test_scaled, test_df_y)}')


Result:

    The Training Logistic Regression Score for the scaled data is 0.713136288998358
    The Testing Logistic Regression Score for the scaled data is 0.893236920459379

This is the code for the `RandomForestClassifier` model that was reapplied on the scaled data. 

    rlf_train_scaled = RandomForestClassifier(random_state=1, n_estimators=500).fit(X_train_scaled, train_df_y)
    print(f'Random Forest Classifier model score for the scaled training data is {rlf_train_scaled.score(X_train_scaled, train_df_y)}')
    print(f'Random Forest Classifier model score for the scaled testing data is {rlf_train_scaled.score(X_test_scaled, test_df_y)}')


Result:

    Random Forest Classifier model score for the scaled training data is 1.0
    Random Forest Classifier model score for the scaled testing data is 0.6193109315185028

## Classification Report

A classification report was preparied which is provided in the code.  This found that the test data was highly sensitive and was not precise which seems to conclude that the data is insufficient enough to make it stable.  Further, it includes a lot of false positives and true negatives.  

              precision    recall  f1-score   support

           0       0.51      0.24      0.33      2351
           1       0.50      0.77      0.61      2351

    accuracy                           0.51      4702
   macro avg       0.51      0.51      0.47      4702
weighted avg       0.51      0.51      0.47      4702

## Conclusion

#### The model does not seem precise and is very sensitive which again leads me to think that there is insufficient data in this sample.