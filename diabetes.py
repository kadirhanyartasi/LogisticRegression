import pandas as pd
import numpy as np

df=pd.read_csv('diabetes2.csv')

df.head()

df.describe().T

df.columns

df.nunique()

df['Outcome'].value_counts()

print("No of zero in Glucose ",df['Glucose'].isin([0]).sum())
print("No of zero in BloodPressure ",df['BloodPressure'].isin([0]).sum())
print("No of zero in SkinThickness ",df['SkinThickness'].isin([0]).sum())
print("No of zero in Insulin ",df['Insulin'].isin([0]).sum())
print("No of zero in BMI ",df['BMI'].isin([0]).sum())

df_pre=df.copy()

df_pre['Glucose']=df_pre['Glucose'].replace(0,df['Glucose'].mean())
df_pre['BloodPressure']=df_pre['BloodPressure'].replace(0,df['BloodPressure'].mean())
df_pre['SkinThickness']=df_pre['SkinThickness'].replace(0,df['SkinThickness'].mean())
df_pre['Insulin']=df_pre['Insulin'].replace(0,df['Insulin'].mean())
df_pre['BMI']=df_pre['BMI'].replace(0,df['BMI'].mean())

df_pre['Pregnancies'].values[df_pre['Pregnancies'] >1]=1

df_pre.describe().T

import matplotlib.pyplot as plt
import seaborn as sns


mask=np.triu(np.ones_like(df_pre.corr()))
sns.heatmap(df_pre.corr(),cmap='coolwarm',mask=mask,annot=True)
plt.title('Correlation Matrix')
plt.show()

from scipy.stats import pointbiserialr

plt.figure(figsize=(20, 15))

plt.subplot(3,3,1)
sns.scatterplot(data=df_pre, x="BMI", y="SkinThickness")
plt.title('BMI Vs Skinthickness')

plt.subplot(3,3,2)
sns.scatterplot(data=df_pre, x="BloodPressure", y="Age")
plt.title('Age Vs BloodPressure')

plt.subplot(3,3,3)
sns.scatterplot(data=df_pre, x="Glucose", y="Insulin")
plt.title('Insulin Vs Glucose')

plt.show()

#Defining the Quartiles for removal of outliers
Q1=df_pre['BMI'].quantile(0.25)
Q3=df_pre['BMI'].quantile(0.75)
IQR=Q3-Q1
lowoutlier=Q1-1.5*IQR
highoutlier=Q3+1.5*IQR
totaloutlier=((df_pre['BMI']<lowoutlier)|(df_pre['BMI']>highoutlier)).sum()
totaloutlier

# Removal of Outliers
df_pre1=df_pre[(df_pre['BMI']<highoutlier)&(df_pre['BMI']>lowoutlier)]
#validating the removal of outlier
totaloutlier = ((df_pre1['BMI'] < lowoutlier) | (df_pre1['BMI'] > highoutlier)).sum()
print("Total Number of Outliers in the BMI are {}".format(totaloutlier))



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Save x & y variables
y=df_pre1['Outcome']
x=df_pre1.drop('Outcome',axis=1)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Verileri train ve test setlerine bölelim
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=101)

# Logistic Regression modelini oluşturalım ve eğitelim
model = LogisticRegression(max_iter=1000, solver='lbfgs')  # max_iter'i artırarak ve solver belirterek düzeltilmiş
model.fit(x_train, y_train)

# Tahminleri yapalım
y_pred = model.predict(x_test)

from sklearn import metrics
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
disp=metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_predictions(y_test, y_pred)
plt.show()
