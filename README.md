# Adult-income-
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('/content/adult_census_data.csv')
df
df.head()
df.tail()
df.columns
df.shape
df.info()
df.describe()
df.duplicated().sum()
df.drop_duplicates(inplace = True)
df.workclass.value_counts()
df['workclass'].replace(to_replace = ['?','Self-emp-not-inc','Without-pay','Never-worked'], value = 'no-income',inplace = True)
df['workclass'].replace(to_replace = ['Local-gov','State-gov','Federal-gov'], value = 'gov',inplace = True)
df['workclass'].replace(to_replace = 'Self-emp-inc', value = 'Self', inplace = True)
df.workclass.value_counts()
df.workclass.value_counts()
df['sex'].value_counts()
df['native.country'].value_counts()
df['occupation'].value_counts()
df.education.value_counts()
df['education'].replace('Preschool', 'dropout',inplace=True)
df['education'].replace('10th', 'dropout',inplace=True)
df['education'].replace('11th', 'dropout',inplace=True)
df['education'].replace('12th', 'dropout',inplace=True)
df['education'].replace('1st-4th', 'dropout',inplace=True)
df['education'].replace('5th-6th', 'dropout',inplace=True)
df['education'].replace('7th-8th', 'dropout',inplace=True)
df['education'].replace('9th', 'dropout',inplace=True)
df['education'].replace('HS-Grad', 'HighGrade',inplace=True)
df['education'].replace('HS-grad', 'HighGrad',inplace=True)
df['education'].replace('Some-college', 'CommunityCollege',inplace=True)
df['education'].replace('Assoc-acdm', 'CommunityCollege',inplace=True)
df['education'].replace('Assoc-voc', 'CommunityCollege',inplace=True)
df['education'].replace('Bachelors', 'Bachelors',inplace=True)
df['education'].replace('Masters', 'Masters',inplace=True)
df['education'].replace('Prof-school', 'Masters',inplace=True)
df['education'].replace('Doctorate', 'Doctorate',inplace=True)
df['education'].value_counts()
df['marital.status'].replace('Never-married', 'NotMarried',inplace=True)
df['marital.status'].replace(['Married-AF-spouse'], 'Married',inplace=True)
df['marital.status'].replace(['Married-civ-spouse'], 'Married',inplace=True)
df['marital.status'].replace(['Married-spouse-absent'], 'NotMarried',inplace=True)
df['marital.status'].replace(['Separated'], 'Separated',inplace=True)
df['marital.status'].replace(['Divorced'], 'Separated',inplace=True)
df['marital.status'].replace(['Widowed'], 'Widowed',inplace=True)
df['marital.status'].value_counts()
df.replace('?',np.NaN,inplace=True)
df
plt.pie(df["income"].value_counts(), labels=df["income"].value_counts().index, explode=[0.1, 0],
        autopct='%1.1f%%', colors=["g", "b"], shadow=True)
plt.title("INCOME SIZE", fontsize=20, fontweight="bold")
plt.legend()
plt.figure(figsize=(8,8))
sns.countplot(x='income',hue='workclass',data=df,palette='Dark2')
plt.legend(loc='best')
plt.show()
plt.figure(figsize=(8,8))
sns.countplot(x='income',hue='occupation',data=df,palette='Dark2')
plt.legend(loc='best')
plt.show()
plt.figure(figsize=(16,16))
sns.countplot(x='income',hue='native.country',data=df,palette='Dark2')
plt.legend(loc='best')
plt.show()
plt.figure(figsize=(8,8))
sns.countplot(x='income',hue='marital.status',data=df,palette='Dark2')
plt.legend(loc='best')
plt.show()
plt.figure(figsize=(8,8))
sns.countplot(x='income',hue='relationship',data=df,palette='Dark2')
plt.legend(loc='best')
plt.show()
plt.figure(figsize=(8,8))
sns.countplot(x='income',hue='race',data=df,palette='Dark2')
plt.legend(loc='best')
plt.show()
plt.figure(figsize=(8,8))
sns.countplot(x='income',hue='sex',data=df,palette='magma')
plt.legend(loc='best')
plt.show()
sns.histplot(df.age, kde = True);
plt.title("Distribution of Age");
sns.pairplot(df, diag_kind = 'kde');
sns.heatmap(df.corr(),annot=True)
df.isna().sum()
df['occupation'].fillna(df['occupation'].mode()[0],inplace=True)
df['native.country'].fillna(df['native.country'].mode()[0],inplace=True)
df.isna().sum()
df.dtypes
df.replace (to_replace = ['<=50K', '>50K'], value = [0,1], inplace = True)
df
df1=pd.get_dummies(df[['workclass','education','marital.status','occupation','relationship','race','sex','native.country']],drop_first=True)
df1
df2
x=df2.drop(['income'],axis=1)
x

df2=pd.concat([df,df1],axis=1)
df2
df2.drop(['fnlwgt','capital.gain','capital.loss','education','education.num','workclass','marital.status','occupation','relationship','race','sex','native.country'],axis=1,inplace=True)
df2
y=df2['income']
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)
x_train
from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()

scalar.fit(x_train)
x_train=scalar.fit_transform(x_train)
x_test=scalar.fit_transform(x_test)
x_train
model=KNeighborsClassifier()from sklearn.neighbors import KNeighborsClassifier
model.fit(x_train,y_train)
y_pred1=model.predict(x_test)
y_pred1
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
mat=confusion_matrix(y_test,y_pred1)
mat
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
label=["No","Yes"]
cmd=ConfusionMatrixDisplay(mat,display_labels=label)
cmd.plot()
score=accuracy_score(y_test,y_pred1)
score
report=classification_report(y_test,y_pred1)
print(report)
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)
y_pred2=nb.predict(x_test)
y_pred2
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
mat=confusion_matrix(y_test,y_pred2)
mat
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
label=["No","Yes"]
cmd=ConfusionMatrixDisplay(mat,display_labels=label)
cmd.plot()
score1=accuracy_score(y_test,y_pred2)
score1
report=classification_report(y_test,y_pred2)
print(report)
from sklearn.svm import SVC
sv=SVC()
sv.fit(x_train,y_train)
y_pred3=sv.predict(x_test)
y_pred3
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
mat=confusion_matrix(y_test,y_pred3)
mat
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
label=["No","Yes"]
cmd=ConfusionMatrixDisplay(mat,display_labels=label)
cmd.plot()
score2=accuracy_score(y_test,y_pred3)
score2
report=classification_report(y_test,y_pred3)
print(report)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred4=dt.predict(x_test)
y_pred4
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
mat=confusion_matrix(y_test,y_pred4)
mat
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
label=["No","Yes"]
cmd=ConfusionMatrixDisplay(mat,display_labels=label)
cmd.plot()
score3=accuracy_score(y_test,y_pred4)
score3
report=classification_report(y_test,y_pred4)
print(report)
from sklearn.ensemble import RandomForestClassifier
rd=RandomForestClassifier()
rd.fit(x_train,y_train)
y_pred5=rd.predict(x_test)
y_pred5
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
mat=confusion_matrix(y_test,y_pred5)
mat
score4=accuracy_score(y_test,y_pred5)
score4
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
label=["No","Yes"]
cmd=ConfusionMatrixDisplay(mat,display_labels=label)
cmd.plot()
report=classification_report(y_test,y_pred5)
print(report)
algorithms=['KNN','Naive Bayes','SVM','Decision Tree','Random Forest']
accuracy=[score,score1,score2,score3,score4]
table=pd.DataFrame({"Algorithms":algorithms,"Accuracy Score":accuracy})
table["Accuracy Score"]=table["Accuracy Score"]*100
table
table1=table.sort_values("Accuracy Score",ascending=False)
table1
plt.bar(table1["Algorithms"],table1["Accuracy Score"],color="g")
plt.xticks(rotation=60)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Comparison of Different Models")
import statsmodels.formula.api as smf 

# Load the statsmodels formula API module

# Logit model with interaction terms   
logit_model = smf.logit('income ~ hours_per_week*occupation + hours_per_week*education', data=df).fit()

# Formula: Outcome ~ Predictors*Interactions
# Fit the model

# Summary
print(logit_model.summary())  

# Display full model summary with coefficients, p-values etc.

# Odds Ratios
print(np.exp(logit_model.params))   

# Exponentiate coefficients to get odds ratios

# Predicted probabilities
probs = logit_model.predict(df)

# Calculate predicted probabilities 

# Plot probability versus hours, segmented by occupation
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)

for occ in df['occupation'].unique():
    sub_df = df[df['occupation'] == occ]
    ax.plot(sub_df['hours.per.week'], probs[sub_df.index], label=occ)

ax.set_xlabel('Hours Worked per Week')
ax.set_ylabel('Probability of Income > $50k')
plt.legend();
from imblearn.over_sampling import SMOTE

# Separate features and target
X = df.drop('income', axis=1)  
y = df['income']

# Instantiate SMOTE 
smote = SMOTE(random_state=42)

# Oversample minority class
X_oversampled, y_oversampled = smote.fit_resample(X, y) 

# Print new class distributions
print('Original dataset shape:', Counter(y))
print('Oversampled data shape: ', Counter(y_oversampled))

# Split oversampled data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_oversampled, y_oversampled, test_size=0.3, random_state=42)

# Build model on oversampled data
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model on test data
from sklearn.metrics import precision_recall_fscore_support
y_pred = model.predict(X_test)
print(precision_recall_fscore_support(y_test, y_pred, average='macro'))

