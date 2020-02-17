# --------------
# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report,confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# Load the data
#Loading the Spam data from the path variable for the mini challenge
#Target variable is the 57 column i.e spam, non-spam classes 
df = pd.read_csv(path)
print(df.shape)
X = df.iloc[:, 0:57]
y = df.iloc[:, 57]

# Overview of the data
#print(df.info())
#print(df.describe())

#Dividing the dataset set in train and test set and apply base logistic model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
# Calculate accuracy , print out the Classification report and Confusion Matrix.
accuracy = lr.score(X_test, y_test)
#print(accuracy)

cm = confusion_matrix(y_test, y_pred)
#print(cm)

report = classification_report(y_test, y_pred)
#print(report)
# Copy df in new variable df1
df1 = df 
relation = df.corr()
#print(relation)
# Remove Correlated features above 0.75 and then apply logistic model
newcorr = relation.unstack().sort_values(kind='quicksort')
#print(newcorr)
hcorr = newcorr[(newcorr > 0.75) & (newcorr != 1)]
#print(hcorr)
df1.drop(labels=['0.25','0.31','0.23'], axis = 1, inplace = True)
#print(df1.shape)
# Split the new subset of data and fit the logistic model on training data
X_new = df1.iloc[:, 0:54]
y_new = df1.iloc[:, 54]
X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(X_new, y_new, random_state=0, test_size = 0.2)
lr.fit(X_new_train, y_new_train)
y_new_pred = lr.predict(X_new_test)

# Calculate accuracy , print out the Classification report and Confusion Matrix for new data
new_accuracy = lr.score(X_new_test, y_new_test)
#print(new_accuracy)

cm_new = confusion_matrix (y_new_test, y_new_pred)
#print(cm_new)

report_new = classification_report (y_new_test, y_new_pred)
#print(report_new)

# Apply Chi Square and fit the logistic model on train data use df dataset
Chi_lr = SelectKBest(score_func = chi2, k=54)
X_new_train = Chi_lr.fit_transform(X_new_train,y_new_train)
X_new_test = Chi_lr.transform(X_new_test)

lr.fit(X_new_train, y_new_train)
y_chi_pred = lr.predict(X_new_test)
# Calculate accuracy , print out the Confusion Matrix 
Chi_accuracy = lr.score(X_new_test,y_new_test)
print(Chi_accuracy)

Cm_chi = confusion_matrix(y_new_test, y_chi_pred)
print(Cm_chi)

# Apply Anova and fit the logistic model on train data use df dataset
Ano_lr = SelectKBest(score_func = f_classif, k=54)
X_new_train = Ano_lr.fit_transform(X_new_train, y_new_train)
X_new_test = Ano_lr.transform(X_new_test)
lr.fit(X_new_train,y_new_train)
y_Ano_pred = lr.predict(X_new_test)
# Calculate accuracy , print out the Confusion Matrix 
an_accuracy = lr.score (X_new_test, y_new_test)
print(an_accuracy)

cm_ano = confusion_matrix(y_new_test, y_Ano_pred)
print(cm_ano)
# Apply PCA and fit the logistic model on train data use df dataset
pca = PCA()
X_new_train = pca.fit_transform(X_new_train, y_new_train)
X_new_test = pca.transform(X_new_test)
# Calculate accuracy , print out the Confusion Matrix 
lr.fit(X_new_train,y_new_train)
y_pca_pred = lr.predict(X_new_test)
# Compare observed value and Predicted value
pca_accuracy = lr.score (X_new_test, y_new_test)
print(pca_accuracy)

cm_pca = confusion_matrix(y_new_test, y_pca_pred)
print(cm_pca)
explained_variance = pca.explained_variance_ratio_
print(explained_variance)


