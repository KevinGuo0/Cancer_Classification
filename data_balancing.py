import pandas as pd
import numpy as np
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, RandomOverSampler

# Set seed
np.random.seed(42)

# Method 1：Under-sampling
def balance_undersampling(data):
    majority = data[data["Class"] == "N"]
    minority = data[data["Class"] == "Y"]
    
    majority_downsampled = resample(majority, 
                                    replace=False, 
                                    n_samples=len(minority), 
                                    random_state=42)
    
    balanced_data = pd.concat([minority, majority_downsampled])
    return balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Method 2：ROSE (RandomOverSampler）
def balance_rose(data):
    X = data.drop("Class", axis=1)
    y = data["Class"]

    ros = RandomOverSampler(random_state=123)
    X_res, y_res = ros.fit_resample(X, y)

    balanced_data = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name="Class")], axis=1)
    return balanced_data

# Method 3：SMOTE
def balance_smote(data):
    X = data.drop("Class", axis=1)
    y = data["Class"]

    sm = SMOTE(sampling_strategy="auto", k_neighbors=5, random_state=123)
    X_res, y_res = sm.fit_resample(X, y)

    balanced_data = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name="Class")], axis=1)
    return balanced_data

# Main
def balance_data_sets(data):
    df_us = balance_undersampling(data)
    df_r = balance_rose(data)
    df_s = balance_smote(data)

    print("\n✅ Class distribution after balancing:\n")
    print(f"Under-sampling:\n{df_us['Class'].value_counts()}\n")
    print(f"ROSE (RandomOverSampler approximation):\n{df_r['Class'].value_counts()}\n")
    print(f"SMOTE:\n{df_s['Class'].value_counts()}\n")

    return df_us, df_r, df_s
