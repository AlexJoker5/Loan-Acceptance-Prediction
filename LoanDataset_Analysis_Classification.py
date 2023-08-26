import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import seaborn as sns
import matplotlib.pyplot as plt


df =pd.read_csv('loan_approval_dataset.csv')

df.head()
df.tail()
df.info()
df.isnull().sum()
df.nunique()

#EDA
#Loan Prediction

sns.countplot(x='loan_status',data=df)
df['loan_status'] = df['loan_status'].apply(lambda x: 1 if x == " Approved" else 0)
df['self_employed'] = df['self_employed'].apply(lambda x: 1 if x == " Yes" else 0)
df['education'] = df['education'].apply(lambda x: 1 if x == " Graduate" else 0)
print(df)
approved = df['loan_status'].value_counts()[1]
rejected = df['loan_status'].value_counts()[0]
print("Approved : ",approved)
print("Rejected : ",rejected)

plt.show()

Credit_Score_ranges = [300, 400, 500, 600, 700, 800, 900, 1000]
df['Credit ScoreRange'] = pd.cut(df['cibil_score'], bins=Credit_Score_ranges)
Credit_Score_range_counts = df['Credit ScoreRange'].value_counts().sort_index()

approved_counts = df[df['loan_status'] == 1]['Credit ScoreRange'].value_counts().sort_index()

print('Credit Score Range Counts:')
print(Credit_Score_range_counts)
print('\nAdmitted Counts:')
print(approved_counts)

plt.bar(Credit_Score_range_counts.index.astype(str), Credit_Score_range_counts.values, label='Total Applicants')
plt.bar(approved_counts.index.astype(str), approved_counts.values, label='Approved Applicants')

plt.xlabel('Credit Score Range')
plt.ylabel('Count')
plt.title('Applicant Count vs. Approved Count in Different Credit Score Ranges')
plt.legend()
plt.show()


median_education_by_credit_score = df.groupby('cibil_score')['education'].median()

print("Median : ", median_education_by_credit_score)
print("Max : ",df['education'].max())
print("Min : ",df['education'].min())

graduated = df['education'].value_counts()[1]
ungraduated = df['education'].value_counts()[0]

print("Graduated : ", graduated)
print("Ungraduated : ", ungraduated)

approved_counts = df[df['loan_status'] == 1]['education'].value_counts().sort_index()
print('\nLoan Approved with Education:')
print(approved_counts)
print("Group by : ",df.groupby(['loan_status', 'education'])['education'].count())

sns.countplot(x="education", hue="loan_status", palette="Set3", data=df)
plt.show()


median_annual_income_by_credit_score = df.groupby('cibil_score')['income_annum'].median()

print("Median : ", median_annual_income_by_credit_score)
print("Max : ",df['income_annum'].max())
print("Min : ",df['income_annum'].min())

Annual_Income_ranges = [200000, 1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000]

df['Annual IncomeRange'] = pd.cut(df['income_annum'], bins=Annual_Income_ranges)
Annual_Income_range_counts = df['Annual IncomeRange'].value_counts().sort_index()
approved_counts = df[df['loan_status'] == 1]['Annual IncomeRange'].value_counts().sort_index()

print('Annual Income Range Counts:')
print(Annual_Income_range_counts)
print('\nAdmitted Counts:')
print(approved_counts)


plt.bar(Annual_Income_range_counts.index.astype(str), Annual_Income_range_counts.values, label='Total Applicants')

plt.bar(approved_counts.index.astype(str), approved_counts.values, label='Approved Applicants')

plt.xlabel('Annual Income Range')
plt.ylabel('Count')
plt.title('Applicant Count vs. Approved Count in Different Annual Income Ranges')

plt.legend()
plt.show()


median_loan_amount_by_credit_score = df.groupby('cibil_score')['loan_amount'].median()

print("Median : ", median_loan_amount_by_credit_score)
print("Max : ",df['loan_amount'].max())
print("Min : ",df['loan_amount'].min())

Loan_Amount_ranges = [300000, 1000000, 5000000, 10000000, 15000000, 20000000, 25000000, 30000000, 35000000, 40000000]

df['Loan AmountRange'] = pd.cut(df['loan_amount'], bins=Annual_Income_ranges)
Loan_Amount_range_counts = df['Loan AmountRange'].value_counts().sort_index()
approved_counts = df[df['loan_status'] == 1]['Loan AmountRange'].value_counts().sort_index()

print('Loan Amount Range Counts:')
print(Loan_Amount_range_counts)
print('\nAdmitted Counts:')
print(approved_counts)


plt.bar(Loan_Amount_range_counts.index.astype(str), Loan_Amount_range_counts.values, label='Total Applicants')

plt.bar(approved_counts.index.astype(str), approved_counts.values, label='Approved Applicants')

plt.xlabel('Loan Amount Range')
plt.ylabel('Count')
plt.title('Applicant Count vs. Approved Count in Different Loan Amount Ranges')

plt.legend()
plt.show()

median_loan_term_by_credit_score = df.groupby('cibil_score')['loan_term'].median()

print("Median : ", median_loan_term_by_credit_score)
print("Max : ",df['loan_term'].max())
print("Min : ",df['loan_term'].min())

term_ranges = [0, 2, 6, 10, 15, 20]
df['LoanRange'] = pd.cut(df['loan_term'], bins=term_ranges)
term_range_counts = df['LoanRange'].value_counts().sort_index()
approved_counts = df[df['loan_status'] == 1]['LoanRange'].value_counts().sort_index()

print('Term Range Counts:')
print(term_range_counts)
print('\nApproved Counts:')
print(approved_counts)

plt.bar(term_range_counts.index.astype(str), term_range_counts.values, label='Total Applicants')

plt.bar(approved_counts.index.astype(str), approved_counts.values, label='Approved Applicants')

plt.xlabel('Term Range')
plt.ylabel('Count')
plt.title('Applicant Count vs. Approved Count in Different Term Ranges')
plt.legend()
plt.show()

median_dependent_by_credit_score = df.groupby('cibil_score')['no_of_dependents'].median()

print("Median : ", median_dependent_by_credit_score)
print("Max : ",df['no_of_dependents'].max())
print("Min : ",df['no_of_dependents'].min())
has_dependents_count = df[df['no_of_dependents'] > 0].shape[0]
no_dependents_count = df[df['no_of_dependents'] == 0].shape[0]

print('Has dependent:', has_dependents_count)
print('No dependent:', no_dependents_count)

approved_counts = df.groupby('no_of_dependents')['loan_status'].sum()
total_counts = df['no_of_dependents'].value_counts()
approved_percentage = (approved_counts / total_counts) * 100
for dependent, percentage in approved_percentage.items():
    print(f'no_of_dependents {dependent}: {percentage:.2f}%')

sns.barplot(x='no_of_dependents', y='loan_status', data=df)
plt.xlabel('Dependent')
plt.ylabel('Approved Percentage')
plt.title('Approved Percentage by Dependent')
plt.show()


median_employment_by_credit_score = df.groupby('cibil_score')['self_employed'].median()

print("Median : ", median_employment_by_credit_score)
print("Max : ",df['self_employed'].max())
print("Min : ",df['self_employed'].min())

employed = df['self_employed'].value_counts()[1]
unemployed = df['self_employed'].value_counts()[0]

print("Employed : ", employed)
print("Unemployed : ", unemployed)

approved_counts = df[df['loan_status'] == 1]['self_employed'].value_counts().sort_index()
print('\nLoan Approved with Employment:')
print(approved_counts)

print(df.head())

print(df.head())

columns_to_standardize = ['income_annum', 'loan_amount']
selected_columns = df[columns_to_standardize]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(selected_columns)
scaled_df = pd.DataFrame(scaled_data, columns=columns_to_standardize)
df[columns_to_standardize] = scaled_df
print(df)

X = df[['no_of_dependents','education','self_employed','income_annum','loan_amount','loan_term','cibil_score']]
y = df['loan_status']

X.head()
y.head()
X.info()
y.info()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8 , random_state=1)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


#Perceptron
ppn = Perceptron(max_iter=200,eta0=0.1, random_state=1)
ppn.fit(X_train, y_train)

y_pred_ppn = ppn.predict(X_test)
print('Misclassified examples: %d' % (y_test != y_pred_ppn).sum())

cm = confusion_matrix(y_test, y_pred_ppn)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')

plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.xticks(ticks=[0, 1], labels=['Class 0', 'Class 1'])
plt.yticks(ticks=[0, 1], labels=['Class 0', 'Class 1'])
plt.show()

accuracy_ppn = accuracy_score(y_test, y_pred_ppn)
print("Accuracy PPN : ")
print(accuracy_ppn)

report = classification_report(y_test, y_pred_ppn)
print("Report : ")
print(report)


#Logistic Classification
LR_model = LogisticRegression(max_iter=200, random_state=1, solver='liblinear')
LR_model.fit(X_train,y_train)

y_pred_lr=LR_model.predict(X_test)
print('Misclassified examples: %d' % (y_test != y_pred_lr).sum())

cm = confusion_matrix(y_test, y_pred_lr)

sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')

plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.xticks(ticks=[0, 1], labels=['Class 0', 'Class 1'])
plt.yticks(ticks=[0, 1], labels=['Class 0', 'Class 1'])
plt.show()

accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(accuracy_lr)

report = classification_report(y_test, y_pred_lr)
print(report)


#Support Vector Machine
print("SVM start")
svm = SVC(kernel='linear', C=1.0, random_state=1)
print("SVM runs step 1")
svm.fit(X_train, y_train)
print("SVM runs step 2")


y_pred_svm=svm.predict(X_test)
print("SVM runs step 3")
print('Misclassified examples: %d' % (y_test != y_pred_svm).sum())

cm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')

# Set labels, title, and ticks
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.xticks(ticks=[0, 1], labels=['Class 0', 'Class 1'])
plt.yticks(ticks=[0, 1], labels=['Class 0', 'Class 1'])
plt.show()

accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(accuracy_svm)
report = classification_report(y_test, y_pred_svm)
print(report)
print("SVM end")


#Model Evaluation

from sklearn.metrics import mean_squared_error

rmse_ppn = np.sqrt(mean_squared_error(y_test, y_pred_ppn))
print("RMSE of Perceptron",rmse_ppn)
      
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
print("RMSE of Logistic Regression",rmse_lr)
      
rmse_svm = np.sqrt(mean_squared_error(y_test, y_pred_svm))
print("RMSE of SVM",rmse_svm)

accuracy = {"Perceptron":accuracy_ppn,"Logistic":accuracy_lr,
            "SVM":accuracy_svm}
rmse={"Perceptron":rmse_ppn,"Logistic":rmse_lr,
            "SVM":rmse_svm}

#Accuracy
labels = list(accuracy.keys())
values = list(accuracy.values())

plt.bar(labels, values)

plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Bar Plot of Accuracy')
plt.show()



#RMSE
labels = list(rmse.keys())
values = list(rmse.values())

plt.bar(labels, values)
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.title('Bar Plot of Error')

plt.show()

import joblib
joblib_file = "Loan_LR_Model.pkl"  
joblib.dump(svm, joblib_file)

#Loan_LR_Model = joblib.load("Loan_LR_Model.pkl")
