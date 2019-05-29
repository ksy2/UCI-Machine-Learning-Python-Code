# Kaelan Yu

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from sklearn.metrics import *
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 155)


# Load Data set
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data"

# Create Pandas Data Frame
df = pd.read_csv(url, header = None)

# Header Names
header_names = ["Class", "Age", "Sex", "Steroid", "Antivirals", "Fatigue", 
              "Malaise", "Anorexia", "Liver Big", "Liver Firm", "Spleen Palpable",
              "Spiders", "Ascites", "Varices", "Bilirubin", "ALP", "SGOT", 
              "Albumin", "Protime", "Histology"]

df.columns = header_names

# Median Imputation of Missing Values
def fillMissingValues():
    for x in range(20):
        attribute = header_names[x]
        df.loc[:, attribute] = pd.to_numeric(df.loc[:, attribute], errors = 'coerce')
        HasNan = np.isnan(df.loc[:, attribute])
        df.loc[HasNan, attribute] = np.nanmedian(df.loc[:, attribute])

fillMissingValues()

# Replace outliers for approximately normally distributed attributes
# with median
def replaceOutliersNormal(attribute):
    mean = np.mean(df.loc[:, attribute])
    median = np.median(df.loc[:, attribute])
    sd = np.std(df.loc[:, attribute])
    lower = mean - 2 * sd
    upper = mean + 2 * sd
    outlier = (df.loc[:, attribute] < lower) | (df.loc[:, attribute] > upper)
    df.loc[outlier, attribute] = median
    return df.loc[:, attribute]
    
# Approximately Normally Distributed Variables
df.loc[:, "ALP"] = replaceOutliersNormal("ALP")
df.loc[:, "Albumin"] = replaceOutliersNormal("Albumin")
df.loc[:, "Protime"] = replaceOutliersNormal("Protime") 

# Replace outliers for approximately exponentially distributed attributes with
# median. For each attribute, values after a certain cutoff point
# are considered outliers.

# Outliers have values > 6 for Bilirubin
median = np.median(df.loc[:, "Bilirubin"])
outlier = df.loc[:, "Bilirubin"] > 6
df.loc[outlier, "Bilirubin"] = median
# Outliers have values > 350 for SGOT
median = np.median(df.loc[:, "SGOT"])
outlier = df.loc[:, "SGOT"] > 350
df.loc[outlier, "SGOT"] = median

# Restore Data Types

# Integer Attributes
integer = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 19]

# Restore all integer variables
def restoreDataType():
    for x in integer:
        attribute = header_names[x]
        df.loc[:, attribute] = df.loc[:, attribute].astype(int)
        
restoreDataType()


# Z-Normalization of Numeric Values
def z_normalize(X):
    return (X - np.mean(X))/np.std(X)
    
df.loc[:, "ALP"] = z_normalize(df.loc[:, "ALP"])
df.loc[:, "Albumin"] = z_normalize(df.loc[:, "Albumin"])
df.loc[:, "Protime"] = z_normalize(df.loc[:, "Protime"])
df.loc[:, "Bilirubin"] = z_normalize(df.loc[:, "Bilirubin"])
df.loc[:, "SGOT"] = z_normalize(df.loc[:, "SGOT"])


# Decoding Categorical Variables

# Class
df.loc[df.loc[:, "Class"] == 1, "Class"] = "die"
df.loc[df.loc[:, "Class"] == 2, "Class"] = "live"

# Sex
df.loc[df.loc[:, "Sex"] == 1, "Sex"] = "male"
df.loc[df.loc[:, "Sex"] == 2, "Sex"] = "female"

# Yes/No Categorical Variables - convert to binary(0/1) dummy variables
for x in range(3, 14):
    attribute = header_names[x]
    df.loc[:, attribute] = df.loc[:, attribute] - 1

# Histology
df.loc[:, "Histology"] = df.loc[:, "Histology"] - 1


# Bin Categorical Variable (Albumin)

# Number of Bins for Albumin (1 = low, 2 = medium, 3 = high)
B = 3
x = df.loc[:, "Albumin"]
bounds = np.linspace(np.min(x), np.max(x), B + 1)

# Binning Function
def bin(x, b): 
    nb = len(b)
    N = len(x)
    y = np.empty(N, int)     
    for i in range(1, nb): 
        y[(x >= bounds[i-1])&(x < bounds[i])] = i    
    y[x == bounds[-1]] = nb - 1 
    return y

binned = bin(x, bounds)
df.loc[:, "Albumin"] = binned

# Creation of New Categorical Variables

# The attribute 'Sex', with the values male or female, is converted to a 
# new binary attribute, 'Male' (0 = female, 1 = male). 
df.loc[:, "Male"] = (df.loc[:, "Sex"] == "male").astype(int)

# The attribute 'Class', with the values die or live, is converted to a 
# new binary attribute, 'dead' (0 = alive, 1 = dead). 
df.loc[:, "Dead"] = (df.loc[:, "Class"] == "die").astype(int)


# Removal of Obsolete Columns
df = df.drop("Class", axis = 1)
df = df.drop("Sex", axis = 1)


###############################################################################
# Split data set 

# Reproducible Results
np.random.seed(seed = 1)

# n = number of observations in the training set
# Target variable = Dead (0 = alive, 1 = dead)
def split_dataset(data, n):     
    
    # Randomize data set
    data = data.sample(frac = 1)    
    
    # Generate training and test sets
    train_X = data.iloc[:n, :-1]
    test_X = data.iloc[n:, :-1]
    train_Y = data.iloc[:n, -1]
    test_Y = data.iloc[n:, -1]
    
    return train_X, test_X, train_Y, test_Y

# Training set of 100 observations, test set of 55 observations
train_X, test_X, train_Y, test_Y = split_dataset(df, 100)


###############################################################################
# Train classifiers

# Random Forest Classifier

# The random forest classifier uses 100 trees, the entropy criterion, and
# requires at least 2 samples to split an internal node.
rfc = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', 
                             min_samples_split = 2)
rfc.fit(train_X, train_Y)

# Support Vector Classifier

# The support vector classifier uses the radial basis function kernel and a
# stopping criterion tolerance of 1e-5.
svc = SVC(kernel = 'rbf', tol = 1e-5, probability = True)
svc.fit(train_X, train_Y)


###############################################################################
# Apply classifiers

# Random Forest Classifier

print("Random Forest Classifier")
print()

# Predicted target values using RFC
print("Predicted target values in test set")
rfc_predicted_values = rfc.predict(test_X)
rfc_predicted_probabilities = rfc.predict_proba(test_X)[:, 1]
print(rfc_predicted_values)
print()

# Support Vector Classifier

print("Support Vector Classifier")
print()

# Predicted target values using SVC
print("Predicted target values in test set")
svc_predicted_values = svc.predict(test_X)
svc_predicted_probabilities = svc.predict_proba(test_X)[:, 1]
print(svc_predicted_values)
print()

# Actual values
print("Actual target values in test set")
actual_values = test_Y.values
print(actual_values)
print() 


###############################################################################
# Measure Classifier Performance

# Random Forest Classifier

print("Random Forest Classifier")
print()

# RFC confusion matrix
rfc_CM = confusion_matrix(actual_values, rfc_predicted_values)
print("Confusion Matrix")
print(rfc_CM)
print()

# RFC error rate
rfc_ER = 1 - accuracy_score(actual_values, rfc_predicted_values)
print("Error Rate: ", round(rfc_ER, 2))
print()

# RFC F1 score
rfc_F1 = f1_score(actual_values, rfc_predicted_values)
print("F1 Score: ", round(rfc_F1, 2))
print()
    
# RFC AUC score
rfc_FPR, rfc_TPR, rfc_TH = roc_curve(actual_values, rfc_predicted_probabilities)
rfc_AUC = auc(rfc_FPR, rfc_TPR)
print("AUC Score: ", np.round(rfc_AUC, 2))
print()


# Support Vector Classifier
print("Support Vector Classifier")
print()

# SVC confusion matrix
svc_CM = confusion_matrix(actual_values, svc_predicted_values)
print("Confusion Matrix")
print(svc_CM)
print()

# SVC error rate
svc_ER = 1 - accuracy_score(actual_values, svc_predicted_values)
print("Error Rate: ", round(svc_ER, 2))
print()

# SVC F1 score
svc_F1 = f1_score(actual_values, svc_predicted_values)
print("F1 Score: ", round(svc_F1, 2))
print()

# SVC AUC score
svc_FPR, svc_TPR, svc_TH = roc_curve(actual_values, svc_predicted_probabilities)
svc_AUC = auc(svc_FPR, svc_TPR)
print("AUC Score: ", np.round(svc_AUC, 2))
print()


# ROC curve plot with RFC, SVC, and random classifier
plt.figure()
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.plot(rfc_FPR, rfc_TPR, color = 'red', lw = 2, label = "Random Forest Classifier")
plt.plot(svc_FPR, svc_TPR, color = 'green', lw = 2, label = "Support Vector Classifier")
plt.plot([0, 1], [0, 1], color = 'blue', lw = 2, linestyle = "--", label = "Random Classifier")
plt.legend(loc = "lower right")
plt.show()
print()