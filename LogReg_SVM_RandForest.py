import pandas as pd

data = pd.read_csv("EDU.csv")

print(data.shape,'\n ', data.size)

print(data.head())
print(data.describe())

data_new = data.loc[:,["gender","raisedhands","VisITedResources","AnnouncementsView","Discussion"]]

data_new.gender = [1 if i == "M" else 0 for i in data_new.gender]

print(data_new.head())

y=data_new.gender.values
x_data=data_new.drop("gender",axis=1) 
print(x_data.head())

import numpy as np
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=52)

from sklearn.linear_model import LogisticRegression
linreg = LogisticRegression()
linreg.fit(x_train,y_train)
print("test accuracy is {}".format(linreg.score(x_test,y_test)))

from sklearn.svm import SVC
svm = SVC(random_state=1)
svm.fit(x_train,y_train)
print("accuracy of svm algorithm ",svm.score(x_test,y_test))


from sklearn.tree import DecisionTreeClassifier
dectree = DecisionTreeClassifier()

dectree.fit(x_train,y_train)
print("accuracy in decision tree is ",dectree.score(x_test,y_test))

from sklearn.ensemble import RandomForestClassifier

ranfor= RandomForestClassifier(n_estimators = 200)
ranfor.fit(x_train,y_train)
print("accuracy of random forest classifier is ",ranfor.score(x_test,y_test))

#confusion matrix

y_pred = ranfor.predict(x_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)

import matplotlib.pyplot as plt
import seaborn as sns
f,ax=plt.subplots(figsize=(8,8))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="blue",fmt=".0f",ax=ax)
plt.xlabel("predicted value")
plt.ylabel("real value")
plt.show()
