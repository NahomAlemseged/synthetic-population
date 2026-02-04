import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from func import *
from scipy.spatial.distance import jensenshannon

# Feature Importance Plot
def plot_feature_importance(model, feature_names, top_n=15):
    """
    Plot the top_n most important features from a fitted XGBClassifier model.
    
    Parameters:
    - model: fitted XGBClassifier
    - feature_names: list of feature names
    - top_n: number of features to plot
    """
    # Get feature importance scores
    importance = model.feature_importances_
    
    # Sort by importance
    indices = np.argsort(importance)[::-1][:top_n]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importance[indices][::-1], align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices][::-1])
    plt.xlabel("Feature Importance Score")
    plt.title(f"Top {top_n} Feature Importances (XGBoost)")
    plt.tight_layout()
    plt.show()

def value_count_icd(data)->pd.DataFrame:
    xx = icd_map(data['APR_MDC'].value_counts().index)
    yy = data['APR_MDC'].value_counts().values
    return pd.DataFrame({'icd_10':xx, 'values':yy})

# Dictionary mapping APR-MDC numbers to category names
def icd_map(list_of_groups):
    apr_mdc_map = {
        1: "Nervous System",
        2: "Eye",
        3: "Ear, Nose, Mouth, Throat",
        4: "Respiratory System",
        5: "Circulatory System",
        6: "Digestive System",
        7: "Hepatobiliary System & Pancreas",
        8: "Musculoskeletal System & Connective Tissue",
        9: "Skin, Subcutaneous Tissue & Breast",
        10: "Endocrine, Nutritional & Metabolic",
        11: "Kidney & Urinary Tract",
        12: "Male Reproductive System",
        13: "Female Reproductive System",
        14: "Pregnancy, Childbirth & Puerperium",
        15: "Newborn & Other Neonates",
        16: "Blood & Blood Forming Organs, Immunological",
        17: "Myeloproliferative Diseases & Neoplasms",
        18: "Infectious & Parasitic Diseases",
        19: "Mental Diseases & Disorders",
        20: "Alcohol/Drug Use & Mental Disorders",
        21: "Injuries, Poisonings, Toxic Effects",
        22: "Burns",
        23: "Other Factors Influencing Health Status",
        24: "Multiple Significant Trauma",
        25: "Human Immunodeficiency Virus Infections"
    }
    return [[x,apr_mdc_map[x]] for x in list_of_groups]
icd_map(list_of_groups = [1,3,4])


def dist_match(data, pred):
    plt.figure(figsize=(8,5))
    plt.hist(data, bins=30, alpha=0.5, label="True", density=True)
    plt.hist(pred, bins=30, alpha=0.5, label="Predicted", density=True)

    plt.xlabel("Class")
    plt.ylabel("Density")
    plt.title("Overlay of Predictions vs True Labels")
    plt.legend()
    plt.show()
    sim = (1-jensenshannon(data,pred)) *1000000
    # sim = 100000
    print(f'Jenson Shannon based similarity: Similarity between actual and synthetic datasets is {sim} percent')

    return sim