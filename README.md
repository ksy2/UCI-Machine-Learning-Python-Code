# UCI Machine Learning Python Code

Summary: Trained random forest and support vector classifiers on the Hepatitis .csv data set from the UCI Machine Learning Repository to make predictions about the patient survival status.

Performed imputation of missing values using the median value for quantitative variables. 

Discarded data values that were considered outliers based on the shape of their distributions (e.g. normal distribution or exponential distribution). 

Performed z-normalization of quantitative variables and converted categorical variables to binary dummy variables in preparation for classification. 

Binned categorical variables such as albumin.

Out of 155 total observations, trained the two classifiers on 100 observations and tested the classifier performance on the remaining 55 observations. 

Measured classifier performance by constructing ROC curves and analyzing performance metrics (error rate, F1 score, and AUC score) derived from a confusion matrix.
