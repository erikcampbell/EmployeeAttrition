"""
Data Translation Challenge- Fewer Features (top-10 related to target & Unbalanced)
Erik C
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier

# Prevent future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load raw data
data= pd.read_excel('employee.xlsx')

# Evaluate for missing values, data types and get description of data
print(data.isnull().any())
print(data.dtypes)
print(data.describe())

# Exploratory Data Analysis (Basic)- all Current/Former Employees
data_hist = data.hist(figsize=(18,18))
print(data_hist)

columns = data.columns.tolist()

# Exploratory data analysis of continuous variables- all current/former employees
fig, ax = plt.subplots(5,2, figsize=(9,9))
sns.distplot(data['TotalWorkingYears'], ax = ax[0,0])
sns.distplot(data['MonthlyIncome'], ax = ax[0,1])
sns.distplot(data['YearsAtCompany'], ax = ax[1,0])
sns.distplot(data['DistanceFromHome'], ax = ax[1,1])
sns.distplot(data['YearsWithCurrManager'], ax = ax[2,0])
sns.distplot(data['YearsSinceLastPromotion'], ax = ax[2,1])
sns.distplot(data['PercentSalaryHike'], ax = ax[3,0])
sns.distplot(data['Age'], ax = ax[3,1])
sns.distplot(data['YearsSinceLastPromotion'], ax = ax[4,0])
sns.distplot(data['TrainingTimesLastYear'], ax = ax[4,1])
plt.tight_layout()
plt.show()

# Current/Past Employees kernel density estimation for numerical values and target - Attrition
plt.figure(figsize=(16,6))
plt.style.use('classic')
plt.grid(True, alpha=0.5)
sns.kdeplot(data.loc[data['Attrition'] == 'No', 'Age'], label = 'Current Employee')
sns.kdeplot(data.loc[data['Attrition'] == 'Yes', 'Age'], label = 'Ex-Employees')
plt.xlim(left=18, right=60)
plt.xlabel('Age (years)')
plt.ylabel('Density')
plt.title('Age Distribution in Percent by Employment Status, All- Range = 18-16, mean =36.9 and standard deviation = 9.1')
plt.legend()

# Commmute distance EDA
plt.figure(figsize=(16,6))
plt.style.use('classic')
plt.grid(True, alpha=0.5)
sns.kdeplot(data.loc[data['Attrition'] == 'No', 'DistanceFromHome'], label = 'Current Employee')
sns.kdeplot(data.loc[data['Attrition'] == 'Yes', 'DistanceFromHome'], label = 'Ex-Employees')
plt.xlim(left=0, right=40)
plt.xlabel('Commute Distance')
plt.ylabel('Density')
plt.title('Work Commute (miles) Distribution in Percent by Employment Status.  All- Range = 1-to-29,  mean= 9.2, standard deviation = 8.1')
plt.legend()

# Years at Company
plt.figure(figsize=(16,6))
plt.style.use('classic')
plt.grid(True, alpha=0.5)
sns.kdeplot(data.loc[data['Attrition'] == 'No', 'YearsAtCompany'], label = 'Current Employee')
sns.kdeplot(data.loc[data['Attrition'] == 'Yes', 'YearsAtCompany'], label = 'Ex-Employees')
plt.xlim(left=0, right=40)
plt.xlabel('Number of Years at Company')
plt.ylabel('Density')
plt.title('Number of Years worked at Firm Distribution in Percent by Employment Status. All- Range = 0-40, mean= 7, standard deviation = 6.1')
plt.legend()

# Years current role
plt.figure(figsize=(16,6))
plt.style.use('classic')
plt.grid(True, alpha=0.5)
sns.kdeplot(data.loc[data['Attrition'] == 'No', 'YearsInCurrentRole'], label = 'Current Employee')
sns.kdeplot(data.loc[data['Attrition'] == 'Yes', 'YearsInCurrentRole'], label = 'Ex-Employees')
plt.xlim(left=0, right=20)
plt.xlabel('Number of Years current position/role')
plt.ylabel('Density')
plt.title('Number of Years in current role in Percent by Employment Status. All- Range = 0-18, mean= 4.2, standard deviation = 3.6')
plt.legend()

# Years since last promotion
plt.figure(figsize=(16,6))
plt.style.use('classic')
plt.grid(True, alpha=0.5)
sns.kdeplot(data.loc[data['Attrition'] == 'No', 'YearsSinceLastPromotion'], label = 'Current Employee')
sns.kdeplot(data.loc[data['Attrition'] == 'Yes', 'YearsSinceLastPromotion'], label = 'Ex-Employees')
plt.xlim(left=0, right=15)
plt.xlabel('Number of Years since last promoted')
plt.ylabel('Density')
plt.title('Number of Years since being promoted in Percent by Employment Status. All- Range = 0-15, mean= 2.2 , standard deviation = 3.2')
plt.legend()

# Years with current manager
plt.figure(figsize=(16,6))
plt.style.use('classic')
plt.grid(True, alpha=0.5)
sns.kdeplot(data.loc[data['Attrition'] == 'No', 'YearsWithCurrManager'], label = 'Current Employee')
sns.kdeplot(data.loc[data['Attrition'] == 'Yes', 'YearsWithCurrManager'], label = 'Ex-Employees')
plt.xlim(left=0, right=17)
plt.xlabel('Years Working with Current Manager')
plt.ylabel('Density')
plt.title('Years with Current Manager in Percent by Employment Status.  All- Range = 0-17, mean = 4.1 ,standard deviation = 3.6')
plt.legend()

# Total working years
plt.figure(figsize=(16,6))
plt.style.use('classic')
plt.grid(True, alpha=0.5)
sns.kdeplot(data.loc[data['Attrition'] == 'No', 'TotalWorkingYears'], label = 'Current Employee')
sns.kdeplot(data.loc[data['Attrition'] == 'Yes', 'TotalWorkingYears'], label = 'Ex-Employees')
plt.xlim(left=0, right=40)
plt.xlabel('Total Working Years')
plt.ylabel('Density')
plt.title('Total Working Years in Percent by Employment Status.  All-Range = 0-40, mean = 11.3  ,standard deviation = 7.8 ')
plt.legend()

# Trainings attended last year
plt.figure(figsize=(16,6))
plt.style.use('classic')
plt.grid(True, alpha=0.5)
sns.kdeplot(data.loc[data['Attrition'] == 'No', 'TrainingTimesLastYear'], label = 'Current Employee')
sns.kdeplot(data.loc[data['Attrition'] == 'Yes', 'TrainingTimesLastYear'], label = 'Ex-Employees')
plt.xlim(left=0, right=40)
plt.xlabel('Total Trainings Attended last year')
plt.ylabel('Density')
plt.title('Total Trainings Attended by employee last year in Percent by Employment Status.  All-Mean = 3 ,standard deviation = 1')
plt.legend()

# Education
plt.figure(figsize=(16,6))
plt.style.use('classic')
plt.grid(True, alpha=0.5)
sns.kdeplot(data.loc[data['Attrition'] == 'No', 'Education'], label = 'Current Employee')
sns.kdeplot(data.loc[data['Attrition'] == 'Yes', 'Education'], label = 'Ex-Employees')
plt.xlim(left=1, right=5)
plt.xlabel('1=No College, 2= College, 3= Bachelor, 4= Master, 5= Doctor')
plt.ylabel('Density')
plt.title('Education-Level of All employees in Percent by Employment Status.')
plt.legend()

# Monthly income
plt.figure(figsize=(16,6))
plt.style.use('classic')
plt.grid(True, alpha=0.5)
sns.kdeplot(data.loc[data['Attrition'] == 'No', 'MonthlyIncome'], label = 'Current Employee')
sns.kdeplot(data.loc[data['Attrition'] == 'Yes', 'MonthlyIncome'], label = 'Ex-Employees')
plt.xlim(left=0, right=27000)
plt.xlabel('Monthly Income')
plt.ylabel('Density')
plt.title('Monthly Income (thousands) in Percent by Employment Status.  Mean = $6,502, Median= $4,919, standard deviation = 4708')
plt.legend()

       
# Find top 5 correlations with the target = Attrition and sort
data_trans = data.copy()
data_trans['Target'] = data_trans['Attrition'].apply(
    lambda x: 0 if x == 'No' else 1)
data_trans = data_trans.drop(['Attrition'], axis =1)
correlations = data_trans.corr()['Target'].sort_values()
print('Most Positive Correlations: \n', correlations.tail(6))
print('\nMost Negative Correlations: \n', correlations.head(6))

# Correlation matrix wtih target = Attrition
cor_mat = data_trans.corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)]=False
fig = plt.gcf()
fig.set_size_inches(60,12)
sns.heatmap(data = cor_mat, mask = mask, square = True, annot = True, cbar = True)

# Create dataframe with variables most correlated with target
data_2 = data[['Attrition','PerformanceRating', 'MonthlyRate', 'NumCompaniesWorked',
              'DistanceFromHome', 'TotalWorkingYears', 'JobLevel', 'YearsInCurrentRole',
              'MonthlyIncome', 'Age']]

print(data_2.dtypes)

#### PREPROCESSING ####

# Create a label encoder object
encoded = LabelEncoder()
for col in data_2.columns[1:]:
    if data_2[col].dtype == 'object':
        if len(list(data_2[col].unique())) <= 2:
            encoded.fit(data_2[col])
            data_2[col] = encoded.transform(data[col])
           
# Categorical variables given dummies
data_2 = pd.get_dummies(data_2, drop_first=True)

# feature scaling and tranform data into floats
scaler = MinMaxScaler(feature_range=(0, 10))
HR_col = list(data_2.columns)
HR_col.remove('Attrition_Yes')
for col in HR_col:
    data_2[col] = data_2[col].astype(float)
    data_2[[col]] = scaler.fit_transform(data_2[[col]])
data_2['Attrition_Yes'] = pd.to_numeric(data_2['Attrition_Yes'], downcast='float')
print(data_2.head())

target = data_2['Attrition_Yes']

# Split df into training and test- 
# trying stratify on Attrition to maintain ratios in split to train/test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    data_2,target, test_size = 0.3, random_state = 7, stratify = target)

print('Number transactions x_train dataset',x_train.shape)
print('Number transactions x_test dataset', x_test.shape)
print('Number transactions y_train dataset',y_train.shape)
print('Number transactions y_test dataset',y_test.shape)

# Logistic Regression
LRmodel = LogisticRegression(class_weight = 'balanced', max_iter=1000)
LRmodel.fit(x_train,y_train)
y_predictions = LRmodel.predict(x_test)
LR_mse = np.sqrt(mean_squared_error(y_test, y_predictions))
LR_confuse_matrix = confusion_matrix(y_test,y_predictions)
print("Logistic Regression Confusion Matrix:\n", LR_confuse_matrix)
print("Logistic Regression Accuracy: ", accuracy_score(y_test, y_predictions))
print("LR MSE: %f" % (LR_mse))
print("LR",classification_report(y_test, y_predictions))

# create confusion matrix visual for LR classifier
LR_cf_matrix = confusion_matrix(y_test,y_predictions)
group_names = ['True Neg', 'False Pos', 'False Neg','True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in LR_cf_matrix.flatten()]
group_percentages = ['{0:.1%}'.format(value) for value in
                     LR_cf_matrix.flatten()/np.sum(LR_cf_matrix)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
LR_cf_matrix = sns.heatmap(LR_cf_matrix, annot=labels, fmt='', cmap='Reds')
plt.title('Logistic Regression Confusion Matrix')

# KNN Classifier
KNN_model = KNeighborsClassifier()
KNN_model.fit(x_train,y_train)
KNN_model_predictions = KNN_model.predict(x_test)
mse = np.sqrt(mean_squared_error(y_test, KNN_model_predictions))
KNN_confuse_matrix = confusion_matrix(y_test,KNN_model_predictions)
print("KNN Confusion Matrix:\n", KNN_confuse_matrix)
print("KNN Accuracy: ", accuracy_score(y_test, KNN_model_predictions))
print("KNN MSE: %f" % (mse))
print(classification_report(y_test, KNN_model_predictions))
plt.title('KNN Confusion Matrix')

# create confusion matrix visual for KNN classifier
group_names = ['True Neg', 'False Pos', 'False Neg','True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in KNN_confuse_matrix.flatten()]
group_percentages = ['{0:.1%}'.format(value) for value in
                      KNN_confuse_matrix.flatten()/np.sum(KNN_confuse_matrix)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
KNN_cf_matrix = sns.heatmap(KNN_confuse_matrix, annot=labels, fmt='', cmap='Reds')

# SVM
SVM_model = SVC()
SVM_model.fit(x_train,y_train)
SVM_model_predictions = SVM_model.predict(x_test)
mse = np.sqrt(mean_squared_error(y_test, SVM_model_predictions))
SVM_confuse_matrix = confusion_matrix(y_test,SVM_model_predictions)
print("SVM Confusion Matrix:\n", SVM_confuse_matrix)
print("SVM Accuracy: ", accuracy_score(y_test, SVM_model_predictions))
print("SVM MSE: %f" % (mse))
print(classification_report(y_test, SVM_model_predictions))

# create confusion matrix visual for SVM classifier
SVM_cf_matrix = confusion_matrix(y_test,SVM_model_predictions)
group_names = ['True Neg', 'False Pos', 'False Neg','True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in SVM_cf_matrix.flatten()]
group_percentages = ['{0:.1%}'.format(value) for value in
                     SVM_cf_matrix.flatten()/np.sum(SVM_cf_matrix)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
SVM_cf_matrix = sns.heatmap(SVM_cf_matrix, annot=labels, fmt='', cmap='Reds')
plt.title('SVM Confusion Matrix')

# Decion Tree Classifier
DTclf = DecisionTreeClassifier(max_depth=2)
DTclf.fit(x_train, y_train)
DT_model_predictions = DTclf.predict(x_test)
DT_mse = np.sqrt(mean_squared_error(y_test, DT_model_predictions))
DT_confuse_matrix = confusion_matrix(y_test, DT_model_predictions)
print('Decision Tree confusion matrix: \n', DT_confuse_matrix)
print("DT Accuracy: ", accuracy_score(y_test, DT_model_predictions))
print("Decision Tree MSE: %f" % (DT_mse))
print("DT",classification_report(y_test, DT_model_predictions))

# create confusion matrix for DT classifier
DT_cf_matrix = confusion_matrix(y_test,DT_model_predictions)
group_names = ['True Neg', 'False Pos', 'False Neg','True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in DT_cf_matrix.flatten()]
group_percentages = ['{0:.1%}'.format(value) for value in
                     DT_cf_matrix.flatten()/np.sum(DT_cf_matrix)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
DT_cf_matrix = sns.heatmap(DT_cf_matrix, annot=labels, fmt='', cmap='Reds')
plt.title('Decision Tree Confusion Matrix')

# selection of algorithms to consider and set performance measure
models = []
models.append(('LR', LogisticRegression(solver='liblinear', random_state=7, class_weight='balanced')))
models.append(('SVM', SVC(gamma='auto', random_state=7)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier(max_depth=2)))

# Generate lists for results
acc_results = []
auc_results = []
names = []

# create table for reporting
col = ['Algorithm', 'ROC AUC Mean', 'ROC AUC STD', 
       'Accuracy Mean', 'Accuracy STD']
stats = pd.DataFrame(columns=col)

i = 0

# evaluate each model using cross-validation
for name, model in models:
    kfold = model_selection.KFold(
        n_splits=10, random_state=7,shuffle=True)  # 10-fold cross-validation

    cv_acc_results = model_selection.cross_val_score(  # accuracy scoring
        model, x_train, y_train, cv=kfold, scoring='accuracy')

    cv_auc_results = model_selection.cross_val_score(  # roc_auc scoring
        model, x_train, y_train, cv=kfold, scoring='roc_auc')

    acc_results.append(cv_acc_results)
    auc_results.append(cv_auc_results)
    names.append(name)
    stats.loc[i] = [name,
                         round(cv_auc_results.mean(), 2),
                         round(cv_auc_results.std(), 2),
                         round(cv_acc_results.mean(), 2),
                         round(cv_acc_results.std(), 2)
                         ]
    i += 1
stats.sort_values(by=['ROC AUC Mean'])
print(stats)