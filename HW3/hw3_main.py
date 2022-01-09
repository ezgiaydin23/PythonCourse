#!/usr/bin/env python
# coding: utf-8

# # HOMEWORK 3 
# # Ezgi Aydın
# The purpose of this homework is to review and practice fundamental machine learning concepts.
# As a first step I will read the data and perform data cleaning, Then feature engineering, variable selection, train-test split, optimization, and model accuracy assessment

# Part 1: Importing Data & Analysis

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import ShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, KFold

import warnings 
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("cses4_cut.csv")


# In[3]:


df.head()


# In[4]:


df.drop("Unnamed: 0", axis=1, inplace=True)


# In[5]:


df.head()


# In[6]:


df.isnull().sum()


# In[7]:


df.shape


# In[8]:


df_observe = df.copy()


# In[9]:


#there seems no null values, however according to the source (https://cses.org/wp-content/uploads/2019/03/cses4_Questionnaire.txt) provided in the document, there are values representing the missing values. So I will first convert those values and drop them. Also will change the column names according to codebook.


# In[10]:


# convert all missing values (ex. 9, 99 or 999) I have performed this part to observe which colums have the most null values but I am excluding it because it breaks my classifiers somehow.
df_observe.loc[df_observe.D2002 > 2, 'D2002'] = np.nan
df_observe.loc[df_observe.D2003 > 96, 'D2003'] = np.nan
df_observe.loc[df_observe.D2004 > 6, 'D2004'] = np.nan
df_observe.loc[df_observe.D2005 > 2, 'D2005'] = np.nan
df_observe.loc[df_observe.D2006 > 2, 'D2006'] = np.nan
df_observe.loc[df_observe.D2007 > 2, 'D2007'] = np.nan
df_observe.loc[df_observe.D2008 > 2, 'D2008'] = np.nan
df_observe.loc[df_observe.D2009 > 2, 'D2009'] = np.nan
df_observe.loc[df_observe.D2010 > 96, 'D2010'] = np.nan
df_observe.loc[df_observe.D2011 > 995, 'D2011'] = np.nan
df_observe.loc[df_observe.D2012 > 6, 'D2012'] = np.nan
df_observe.loc[df_observe.D2013 > 6, 'D2013'] = np.nan
df_observe.loc[df_observe.D2014 > 6, 'D2014'] = np.nan
df_observe.loc[df_observe.D2015 > 96, 'D2015'] = np.nan
df_observe.loc[df_observe.D2016 > 996, 'D2016'] = np.nan
df_observe.loc[df_observe.D2017 > 6, 'D2017'] = np.nan
df_observe.loc[df_observe.D2018 > 6, 'D2018'] = np.nan
df_observe.loc[df_observe.D2019 > 6, 'D2019'] = np.nan
df_observe.loc[df_observe.D2020 > 6, 'D2020'] = np.nan
df_observe.loc[df_observe.D2021 > 96, 'D2021'] = np.nan
df_observe.loc[df_observe.D2022 > 96, 'D2022'] = np.nan
df_observe.loc[df_observe.D2023 > 96, 'D2023'] = np.nan
df_observe.loc[df_observe.D2024 > 6, 'D2024'] = np.nan
df_observe.loc[df_observe.D2025 > 6, 'D2025'] = np.nan
df_observe.loc[df_observe.D2026 > 96, 'D2026'] = np.nan
df_observe.loc[df_observe.D2027 > 996, 'D2027'] = np.nan
df_observe.loc[df_observe.D2028 == 99, 'D2028'] = np.nan
df_observe.loc[df_observe.D2029 > 996, 'D2029'] = np.nan
df_observe.loc[df_observe.D2030 > 996, 'D2030'] = np.nan
df_observe.loc[df_observe.D2031 > 6, 'D2031'] = np.nan


# In[11]:


# Renaming columns in accordance with the codebook
df = df.rename({"D2002": "GENDER", "D2003": "EDUCATION", "D2004": "MARITAL STATUS",
                "D2005": "UNION MEMBERSHIP OF RESPONDENT", "D2006": "UNION MEMBERSHIP OF OTHERS IN HOUSEHOLD",
                "D2007": "BUSINESS OR EMPLOYERS' ASSOCIATION MEMBERSHIP", "D2008": "FARMERS' ASSOCIATION MEMBERSHIP",
                "D2009": "PROFESSIONAL ASSOCIATION MEMBERSHIP", "D2010": "CURRENT EMPLOYMENT STATUS",
                "D2011": "MAIN OCCUPATION", "D2012": "SOCIO ECONOMIC STATUS", "D2013": "EMPLOYMENT TYPE - PUBLIC OR PRIVATE",
                "D2014": "INDUSTRIAL SECTOR", "D2015": "SPOUSE: CURRENT EMPLOYMENT STATUS", "D2016": "SPOUSE: OCCUPATION",
                "D2017": "SPOUSE: SOCIO ECONOMIC STATUS", "D2018": "SPOUSE: EMPLOYMENT TYPE - PUBLIC OR PRIVATE",
                "D2019": "SPOUSE: INDUSTRIAL SECTOR", "D2020": "HOUSEHOLD INCOME", "D2021": "NUMBER IN HOUSEHOLD IN TOTAL",
                "D2022": "NUMBER OF CHILDREN IN HOUSEHOLD UNDER AGE 18", "D2023": "NUMBER IN HOUSEHOLD UNDER AGE 6",
                "D2024": "RELIGIOUS SERVICES ATTENDANCE", "D2025": "RELIGIOSITY", "D2026": "RELIGIOUS DENOMINATION",
                "D2027": "LANGUAGE USUALLY SPOKEN AT HOME", "D2028": "REGION OF RESIDENCE", "D2029": "RACE",
                "D2030": "ETHNICITY", "D2031": "RURAL OR URBAN RESIDENCE", "age": "AGE", "voted": "VOTED"}, axis=1)


# In[12]:


#right below we see which columns have how much null values. I will drop the columns with null values larger than 50% ish. So the first 11 columns below will be dropped.


# In[13]:


df_observe.isnull().sum().sort_values(ascending=False)


# In[14]:


#dropping columns with highest null values


# In[15]:


cleanDf = df.drop(columns =["RELIGIOUS DENOMINATION","SPOUSE: SOCIO ECONOMIC STATUS","SPOUSE: OCCUPATION","ETHNICITY","SPOUSE: EMPLOYMENT TYPE - PUBLIC OR PRIVATE","SPOUSE: INDUSTRIAL SECTOR","MAIN OCCUPATION","SPOUSE: CURRENT EMPLOYMENT STATUS","BUSINESS OR EMPLOYERS' ASSOCIATION MEMBERSHIP","PROFESSIONAL ASSOCIATION MEMBERSHIP","FARMERS' ASSOCIATION MEMBERSHIP"], axis=1)


# In[16]:


cleanDf.head()


# In[17]:


cleanDf.apply(pd.to_numeric, errors='coerce')


# In[18]:


cleanDf.info()


# In[19]:


#Now data is cleaner, I will not manipulate other missing values, we will perform imputation for them on the next steps.


# In[20]:


#splitting train and test data


# In[21]:


X = cleanDf.iloc[:, :-1]
y = cleanDf.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=23)


# In[22]:


#let's see classifier accuracies before any reduction


# In[23]:


cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=2)


# In[24]:


#Logistic Regression
LR = LogisticRegression()
LR_accuracy=cross_val_score(LR, X, y, cv=cv).mean()

#K-Nearest Neighbors
KNN = KNeighborsClassifier()
KNN_accuracy=cross_val_score(KNN, X, y, cv=cv).mean()


#Linear Discriminant Analysis
LDA = LinearDiscriminantAnalysis()
LDA_accuracy=cross_val_score(LDA, X, y, cv=cv).mean()

#Decision Tree
decision_tree = DecisionTreeClassifier()
DT_accuracy=cross_val_score(decision_tree, X, y, cv=cv).mean()


#Support Vector Machine
SVM = SVC(probability = True)
SVM_accuracy=cross_val_score(SVM, X, y, cv=cv).mean()


#Quadratic Discriminant Analysis
QDA = QuadraticDiscriminantAnalysis()
QDA_accuracy=cross_val_score(QDA, X, y, cv=cv).mean()

#Random Forest Classifier
random_forest = RandomForestClassifier()
RF_accuracy=cross_val_score(random_forest, X, y, cv=cv).mean()

#Naive Bayes
naive_bayes = GaussianNB()
NB_accuracy=cross_val_score(naive_bayes, X, y, cv=cv).mean()

pd.options.display.float_format = '{:,.2f}%'.format
accuracies_before = pd.DataFrame({
    'Model'       : ['Logistic Regression', 'K-Nearest Neighbors', 'Linear Discriminant Analysis','Decision Tree', 'Support Vector Machine', 'Quadratic Discriminant Analysis', 'Random Forest','Bayes'],
    'Accuracy'    : [100*LR_accuracy, 100*KNN_accuracy, 100*LDA_accuracy, 100*DT_accuracy, 100*SVM_accuracy, 100*QDA_accuracy, 100*RF_accuracy, 100*NB_accuracy],
    }, columns = ['Model', 'Accuracy'])

accuracies_before = accuracies_before.sort_values(by='Accuracy', ascending=False)


# In[25]:


#feature selection
X.shape

X_new = SelectKBest(chi2, k='all').fit_transform(X, y)

score = SelectKBest(chi2, k='all')
fit = score.fit(X, y)
k_scores = fit.scores_

X_new.shape
k_scores


# In[26]:


X_new


# In[27]:


# Features in descending order by score
features = pd.DataFrame([cleanDf.columns, k_scores])
features = features.T

features.columns
features = features.sort_values(by=features.columns[1], ascending=False)


# In[28]:


highestK_10 = features[:10]


# In[29]:


highestK_10


# In[30]:


highscorer_df = df[['LANGUAGE USUALLY SPOKEN AT HOME',"RACE","NUMBER IN HOUSEHOLD IN TOTAL","NUMBER OF CHILDREN IN HOUSEHOLD UNDER AGE 18","REGION OF RESIDENCE","NUMBER IN HOUSEHOLD UNDER AGE 6","AGE","EMPLOYMENT TYPE - PUBLIC OR PRIVATE","INDUSTRIAL SECTOR","RELIGIOSITY"]]


# In[31]:


highscorer_df


# In[32]:


# data distribution of highest features

plt.figure(figsize = (20, 20))
plotnum = 1

for column in highscorer_df:
    if plotnum <= 12:
        ax = plt.subplot(4, 5, plotnum)
        sns.distplot(highscorer_df[column])
        plt.xlabel(column)
        
    plotnum += 1

plt.tight_layout()
plt.show()


# In[33]:


#so the data is not normally distributed, need to fix that.
#Pre-processing and transforming in Gaussian form.

quantile_transformer = preprocessing.QuantileTransformer(random_state=10)

highscorer_df_tr = quantile_transformer.fit_transform(highscorer_df)


# In[34]:


plt.figure(figsize = (20, 20))
plotnum = 1
for column in range(highscorer_df_tr.shape[1]):
    if plotnum <= 12:
        ax = plt.subplot(4, 5, plotnum)
        sns.distplot(highscorer_df_tr[column])
        plt.xlabel(column)
        
    plotnum += 1

plt.tight_layout()
plt.show()


# In[35]:


#Classifiers after preprocessing and transforming
#Logistic Regression
LR = LogisticRegression()
LR_accuracy=cross_val_score(LR, highscorer_df_tr, y, cv=cv).mean()

#Decision Tree
decision_tree = DecisionTreeClassifier()
DT_accuracy=cross_val_score(decision_tree, highscorer_df_tr, y, cv=cv).mean()

#Support Vector Machine
SVM = SVC(probability = True)
SVM_accuracy=cross_val_score(SVM, highscorer_df_tr, y, cv=cv).mean()

#Linear Discriminant Analysis
LDA = LinearDiscriminantAnalysis()
LDA_accuracy=cross_val_score(LDA, highscorer_df_tr, y, cv=cv).mean()

#Quadratic Discriminant Analysis
QDA = QuadraticDiscriminantAnalysis()
QDA_accuracy=cross_val_score(QDA, highscorer_df_tr, y, cv=cv).mean()

#Random Forest Classifier
random_forest = RandomForestClassifier()
RF_accuracy=cross_val_score(random_forest, highscorer_df_tr, y, cv=cv).mean()

#K-Nearest Neighbors
KNN = KNeighborsClassifier()
KNN_accuracy=cross_val_score(KNN, highscorer_df_tr, y, cv=cv).mean()

#Naive Bayes
bayes = GaussianNB()
NB_accuracy=cross_val_score(bayes, highscorer_df_tr, y, cv=cv).mean()

pd.options.display.float_format = '{:,.2f}%'.format
accuracies_after = pd.DataFrame({'Model'       : ['Logistic Regression', 'K-Nearest Neighbors', 'Linear Discriminant Analysis','Decision Tree', 'Support Vector Machine', 'Quadratic Discriminant Analysis', 'Random Forest','Bayes'],
'Accuracy'    : [100*LR_accuracy, 100*KNN_accuracy, 100*LDA_accuracy, 100*DT_accuracy, 100*SVM_accuracy, 100*QDA_accuracy, 100*RF_accuracy, 100*NB_accuracy],}, columns = ['Model', 'Accuracy'])

accuracies_after = accuracies_after.sort_values(by='Accuracy', ascending=False)


# In[36]:


#to find hyperparameters, used top 5 classifiers


# In[39]:


#Random Forest Classifier

best_score=0
n_estimators= [100,200,500,1000]
criteria=['gini', 'entropy']
for i in n_estimators:
    for k in criteria:
        random_forest = RandomForestClassifier(n_estimators=i,criterion=k)
        RF_accuracy=cross_val_score(random_forest, highscorer_df_tr, y, cv=cv).mean()
        if RF_accuracy > best_score:
            best_score=RF_accuracy
            best_est=i
            best_cri=k
RF_accuracy=best_score
print("Best score:",best_score,"estimator:",best_est,"criterion:",best_cri)


#Support Vector Machine

best_score=0
list=[0.1,1,2,5]
kernel=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed2']
for i in list:
    for k in kernel:
        SVM = SVC(C=i,kernel=k)
        SVM_accuracy=cross_val_score(SVM, highscorer_df_tr, y, cv=cv).mean()
        if SVM_accuracy>best_score:
            best_score=SVM_accuracy
            best_c=i
            best_k=k
SVM_accuracy=best_score
print("Best score:",best_score,"c:",best_c,"kernel:",k)
        

#Linear Discriminant Analysis
        
best_score=0        
solver=['svd', 'lsqr', 'eigen']
for i in solver:    
    LDA = LinearDiscriminantAnalysis(solver=i)
    LDA_accuracy=cross_val_score(LDA, highscorer_df_tr, y, cv=cv).mean()
    if LDA_accuracy>best_score:
        best_score=LDA_accuracy
        best_solver=i
LDA_accuracy=best_score
print("Best score:",best_score,"solver:",best_solver)


#Logistic Regression

best_score=0     
penalty=['l1', 'l2', 'elasticnet', 'none']
for i in penalty:
    LR = LogisticRegression(penalty=i)
    LR_accuracy=cross_val_score(LR, highscorer_df_tr, y, cv=cv).mean()
    if LR_accuracy > best_score:
        best_score=LR_accuracy
        best_p=i
LR_accuracy=best_score
print("Best score:",best_score,"penalty",best_p)



#K-Nearest Neighbors
        
best_score=0
for i in range(2,10):
    KNN = KNeighborsClassifier(n_neighbors=i)
    KNN_accuracy=cross_val_score(KNN, highscorer_df_tr, y, cv=cv).mean()
    if KNN_accuracy > best_score:
        best_score=KNN_accuracy
        best_n=i
KNN_accuracy=best_score
print("Best score:",best_score,"neighbors:",best_n)

pd.options.display.float_format = '{:,.2f}%'.format
accuracies_last = pd.DataFrame({
    'Model'       : ['Logistic Regression', 'Support Vector Machine', 'Linear Discriminant Analysis', 'Random Forest', 'K-Nearest Neighbors',],
    'Accuracy'    : [100*LR_accuracy, 100*SVM_accuracy, 100*LDA_accuracy, 100*RF_accuracy, 100*KNN_accuracy],
    }, columns = ['Model', 'Accuracy'])

accuracies_last=accuracies_last.sort_values(by='Accuracy', ascending=False)


# In[40]:


print("Classifiers before reduction:")
display(accuracies_before)
print("Classifiers with dimensionality-reduction and pre-processing:")
display(accuracies_after)
print("Classifiers with optimized model and its hyperparameters:")
display(accuracies_last)


# In[ ]:




