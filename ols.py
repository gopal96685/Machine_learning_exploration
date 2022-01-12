import pandas as pd
import numpy as np 
import sklearn
from sklearn.model_selection import train_test_split
data = pd.read_csv("head.csv" , header = [0])
data = pd.DataFrame(data)

from sklearn.linear_model import LinearRegression


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


print(data.keys())
print(data.head)

print(data.describe())
print(data.shape)
data=data[['Brain Weight(grams)', 'Head Size(cm^3)']]


##cleaning
data = clean_dataset(data)
data.dropna()
print(np.all(np.isfinite(data)))
print(data.shape)

X = data[['Head Size(cm^3)']]
Y = data[['Brain Weight(grams)']]

print(X.shape)
print(Y.shape)

#splitting into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.002, random_state = 100)

print("X_train shape",X_train.shape)
print("X_test shape",X_test.shape)
print("Y_train shape",Y_train.shape)
print("Y_test shape",Y_test.shape)


lm = LinearRegression()
lm.fit(X_train, Y_train)
Y_pred = lm.predict(X_test)


#score of the model ( r2 score )
print(f'r_sqr value: {lm.score(X_train, Y_train)}')































