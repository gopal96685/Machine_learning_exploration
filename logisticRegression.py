import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("advertising.csv")



def gender_writer(datafr):
	datafr['Sex'] = datafr.apply(lambda row: "Male" if row.Male else "Female", axis = 1)


def timestamp_hasing(datafr):
	datafr['Month'] = datafr['Timestamp'].apply(lambda x: x.split('-')[1])
	datafr['Time'] = datafr['Timestamp'].apply(lambda x: x.split(':')[0].split(' ')[1])


def get_countrywise_viewership(datafr):
	datafr.groupby(by="Country")


# print(dataset)
# print(dataset.keys())
# print(dataset.shape)
# print(dataset.describe())

dataset.dropna()

gender_writer(dataset)
timestamp_hasing(dataset)

X = dataset.drop(['Timestamp', 'Clicked on Ad', 'Ad Topic Line', 'Country', 'City', 'Sex', 'Month', 'Time'], axis=1)
y = dataset['Clicked on Ad']


# plt.figure(figsize=(10,7))
# sns.displot(x = dataset['Age'], bins = 20, kde=True)
# sns.displot( x = dataset['Age'], y = dataset['Area Income'])
# sns.displot( x = dataset['Age'], y = dataset['Daily Time Spent on Site'])
# g = sns.catplot(
#     data=dataset, kind="bar",
#     x="Age", y="Daily Internet Usage", hue="Sex",
#     ci="sd", palette="dark", alpha=.6, height=6
# )

# sns.scatterplot(x=dataset['Age'], y = dataset['Daily Time Spent on Site'], data = dataset )
# g = sns.catplot(data=dataset, kind="bar", y="Daily Time Spent on Site", x="Age")
# plt.show()


from sklearn.metrics import accuracy_score

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        
    elif train==False:
        pred = clf.predict(X_test)
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)

y_pred = lr_clf.predict(X_test)


print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)
print_score(lr_clf, X_train, y_train, X_test, y_test, train=False)
















