#Needed libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.preprocessing import StandardScaler
from sklearn import pipeline
from sklearn import compose
import time


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
print(df.shape)
X, y = df.drop(['target'], axis=1), df['target']


def classifier():
    tree_classifiers = {
    "Decision Tree": DecisionTreeClassifier(),
    "Extra Trees": ExtraTreesClassifier(),
    "Random Forest": RandomForestClassifier(),
    #"AdaBoost": AdaBoostClassifier(),
    #"Skl GBM": GradientBoostingClassifier(),
    #"Skl HistGBM":HistGradientBoostingClassifier(),
    #"XGBoost": XGBClassifier(),
    #"LightGBM": LGBMClassifier(),
    #"CatBoost": CatBoostClassifier()
    }

    #cat_vars  = ['f_07','f_08','f_09','f_10','f_11','f_12','f_13','f_14','f_15','f_16','f_17','f_18', 'f_29', 'f_30' ]
    num_vars  = ['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 'f_06','f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25','f_26', 'f_28']
    num_prepro = pipeline.Pipeline(steps=[('scaler', StandardScaler())])
    #cat = pipeline.Pipeline(steps=[('ordinal', OrdinalEncoder(handle_unknown='error'))])
    tree_pre  = compose.ColumnTransformer(transformers=[('num', num_prepro, num_vars)], remainder='passthrough')

    tree_classifiers = {name: pipeline.make_pipeline(tree_pre, model) for name, model in tree_classifiers.items()}
    return tree_classifiers
classifiers = classifier()


def split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = split(X,y)



def model_results(X_train, X_test, y_train, y_test):

    results = pd.DataFrame({'Model': [], 'Accuracy': [], 'Bal Acc.': [], 'Time': []})

    for model_name, model in classifiers.items():
        start_time = time.time()        
        model.fit(X_train,y_train)
        pred =model.predict(X_test)

        total_time = time.time() - start_time

        results = results.append({"Model":    model_name,
                                "Accuracy": metrics.accuracy_score(y_test, pred)*100,
                                "Bal Acc.": metrics.balanced_accuracy_score(y_test, pred)*100,
                                "Time":     total_time},
                                ignore_index=True)
                                
    results_ord = results.sort_values(by=['Accuracy'], ascending=False, ignore_index=True)
    results_ord.index += 1 
    results_ord.style.bar(subset=['Accuracy', 'Bal Acc.'], vmin=0, vmax=100, color='#5fba7d')
    return results_ord
mod_res = model_results(X_train, X_test, y_train, y_test)
print(mod_res)

#        Model      Accuracy   Bal Acc.        Time
#1  Random Forest  82.442088  82.376508  424.815111
#2    Extra Trees  79.629966  79.541862  188.617821
#3  Decision Tree  72.365657  72.346663   33.526589


model  = classifiers[3]