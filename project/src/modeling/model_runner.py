import pandas as pd
from sklearn.model_selection import cross_val_score

# Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def run_regression(X, y):

    models = {

        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
        "DecisionTree": DecisionTreeRegressor(),
        "RandomForest": RandomForestRegressor(),
        "GradientBoosting": GradientBoostingRegressor(),
        "XGBoost": XGBRegressor(),
        "KNN": KNeighborsRegressor(),
        "SVR": SVR(),
        "AdaBoost": AdaBoostRegressor(),
        "LightGBM": LGBMRegressor(),
        "CatBoost": CatBoostRegressor(verbose=0)

    }

    results = []

    for name, model in models.items():

        score = cross_val_score(model, X, y, cv=5, scoring="r2").mean()

        results.append([name, score * 100])

    return pd.DataFrame(results, columns=["Model", "Accuracy (%)"])

def run_classification(X, y):

    models = {

        "Logistic": LogisticRegression(max_iter=200),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
        "XGBoost": XGBClassifier(),
        "LightGBM": LGBMClassifier(),
        "CatBoost": CatBoostClassifier(verbose=0),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(),
        "NaiveBayes": GaussianNB(),
        "AdaBoost": AdaBoostClassifier(),
        "ExtraTrees": ExtraTreesClassifier()

    }

    results = []

    for name, model in models.items():

        score = cross_val_score(model, X, y, cv=5, scoring="accuracy").mean()

        results.append([name, score * 100])

    return pd.DataFrame(results, columns=["Model", "Accuracy (%)"])
