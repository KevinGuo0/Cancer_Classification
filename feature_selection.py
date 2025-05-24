import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

# 1. PCA
def pca_selection(X, variance_threshold=0.95):
    pca = PCA()
    pca.fit(X)
    
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.searchsorted(cum_var, variance_threshold) + 1
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    print(f"[PCA] Selected {n_components} components to retain {variance_threshold*100:.1f}% variance.")
    return X_pca, n_components, pca


# 2. Information gain
def info_gain_selection(X, y, top_k=10):
    mi = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
    mi_scores = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    
    selected_features = mi_scores.head(top_k).index.tolist()
    
    print(f"[InfoGain] Top {top_k} features:\n{selected_features}")
    return selected_features, mi_scores


# 3. Boruta
def boruta_selection(X, y, n_estimators=100, max_iter=100):
    rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=42)
    boruta_selector = BorutaPy(estimator=rf, n_estimators='auto', max_iter=max_iter, random_state=42)
    boruta_selector.fit(np.array(X), np.array(y))
    
    selected_features = X.columns[boruta_selector.support_].tolist()
    
    print(f"[Boruta] Selected features:\n{selected_features}")
    return selected_features, boruta_selector
