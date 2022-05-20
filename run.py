#Needed libraries
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.impute import  SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn import pipeline
from sklearn import compose
from sklearn.compose import ColumnTransformer


from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from sklearn.ensemble      import ExtraTreesClassifier
from sklearn.ensemble      import AdaBoostClassifier
from sklearn.ensemble      import GradientBoostingClassifier
from sklearn.ensemble      import HistGradientBoostingClassifier
#from xgboost               import XGBClassifier
#from lightgbm              import LGBMClassifier
#from catboost              import CatBoostClassifier

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


def tree_classifiers():
    tree_classifiers = {
    "Decision Tree": DecisionTreeClassifier(),
    "Extra Trees": ExtraTreesClassifier(),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Skl GBM": GradientBoostingClassifier(),
    "Skl HistGBM":HistGradientBoostingClassifier(),
    #"XGBoost": XGBClassifier(),
    #"LightGBM": LGBMClassifier(),
    #"CatBoost": CatBoostClassifier()
    }

    cat_vars  = ['f_', 'Embarked', 'Title']
    num_vars  = ['Pclass', 'SibSp', 'Parch', 'Fare', 'Age']
    num_prepro = pipeline.Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent'))])
    cat = pipeline.Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),('ordinal', OrdinalEncoder(handle_unknown='ignore'))])
    tree_pre  = compose.ColumnTransformer(transformers=[('num', num_prepro, num_vars),('cat', cat, cat_vars)], remainder='drop')

    tree_classifiers = {name: pipeline.make_pipeline(tree_pre, model) for name, model in tree_classifiers.items()}
    return tree_classifiers
