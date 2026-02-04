# Preprocessing
# Replace backticks and empty strings in the entire DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
# import gower
from sklearn.neighbors import NearestNeighbors
from helpers import *
import matplotlib.pyplot as plt
import xgboost as xgb




def preprocess(df):

    df_ = df.copy()
    cat_cols = ['DISCHARGE','TYPE_OF_ADMISSION', 'SOURCE_OF_ADMISSION', 'ILLNESS_SEVERITY',
                'PUBLIC_HEALTH_REGION', 'PAT_STATUS', 'SEX_CODE', 'RACE', 'ETHNICITY', 'ADMIT_WEEKDAY','PAT_AGE']
    for col in cat_cols:
        df_[col] = df_[col].astype(str)
        le = LabelEncoder()
        df_[col] = le.fit_transform(df_[col])
    num_cols = ['LENGTH_OF_STAY']
    X = df_[cat_cols + num_cols]
    y = df_['APR_MDC']
    return X, y


def preprocess_pca(X,n=4):
    # df.replace('`', np.nan, inplace=True)
    # df.replace('', np.nan, inplace=True)
    # df.dropna(inplace=True)
    # from sklearn.decomposition import PCA
    pca = PCA(n_components=7)  # reduce to 2 components
    # cat_cols = ['DISCHARGE','TYPE_OF_ADMISSION', 'SOURCE_OF_ADMISSION', 'ILLNESS_SEVERITY',
    #             'PUBLIC_HEALTH_REGION', 'PAT_STATUS', 'SEX_CODE', 'RACE', 'ETHNICITY', 'ADMIT_WEEKDAY']
    # for col in cat_cols:
    #     df_[col] = df_[col].astype(str)
    #     le = LabelEncoder()
    #     df_[col] = le.fit_transform(df_[col])
    # num_cols = ['LENGTH_OF_STAY', 'PAT_AGE']
    # X = df_[cat_cols + num_cols]
    X_pca = pca.fit_transform(X)
    # y = df_['APR_MDC']

    print("Explained variance:", pca.explained_variance_)
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Sum ratio:", sum(pca.explained_variance_ratio_))
    return X_pca

# Training
def train_model(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


    # model = xgb(n_jobs=-1)
    model.fit(X_train, y_train)
    return model, X_test, y_test




def eval_model(model, X_test, y_test)->pd.DataFrame:
    preds = model.predict(X_test)
    df_pred = pd.DataFrame(X_test).copy()
    df_pred["id"] = X_test.index
    df_pred["y_test"] = y_test
    df_pred["pred"] = preds


    acc = accuracy_score(y_test, preds)
    f1 = classification_report(y_test, preds)
    print("Accuracy:", acc)
    print(f'classification_report.{f1}')
    plot_feature_importance(model, feature_names=X_test.columns, top_n=10)
    return df_pred


