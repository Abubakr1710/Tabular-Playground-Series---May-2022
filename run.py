#Needed libraries
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv('train.csv')
df.drop(['id', 'f_27'], axis=1, inplace=True)
X, y = df.drop(['target'], axis=1), df['target']

def split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = split(X,y)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

