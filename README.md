# UCI Machine Learning Python Code

Summary: Trained random forest and support vector classifiers on the Hepatitis .csv data set from the UCI Machine Learning Repository to make predictions about the patient survival status.

1. Importing the data set/Python libraries

Python libraries used
- numpy
- pandas
- sklearn - to obtain access to the random forest and support vector classifiers
- matplotlib

2. Data cleaning and preparation
- Performed imputation of missing values using the median value for quantitative variables. 
- Discarded data values that were considered outliers based on the shape of their distributions (e.g. normal distribution or exponential distribution). 
- Performed z-normalization of quantitative variables and converted categorical variables to binary dummy variables in preparation for classification. 
- Binned quantitative variables such as albumin to a small number of groups (low, middle, high albumin value).


3. Training machine learning models
- For the random forest classifier: utilized 100 trees, the entropy criterion, and required at least 2 samples to split an internal node.
- For the support vector classifier: used the radial basis function kernel and a stopping criterion tolerance of 1e-5 
- Out of 155 total observations, trained the two classifiers on 100 observations and tested the classifier performance on the remaining 55 observations (approximately a 60/40 training/test set split). 


4. Evaluating model performance
- Measured classifier performance by constructing ROC curves and analyzing performance metrics (error rate, F1 score, and AUC score) derived from the confusion matrix.
