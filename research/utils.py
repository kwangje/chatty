import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def train_test_split(df, test_size=1000):
    conv_ids = df['conv_id'].unique()
    test_indices = np.random.choice(conv_ids, size=test_size, replace=False)
    train_indices = np.setdiff1d(conv_ids, test_indices)
    return df[df['conv_id'].isin(train_indices)], df[df['conv_id'].isin(test_indices)]


def cv_stratified_shuffle(X: np.array,
                          y: np.array,
                          model,
                          splits=5,
                          probability=True):
    """Rusn stratified shuffle split on X, y, with given model, for n splits

    Parameters
    ----------
    X : np.array
    y : np.array
    model : sklearn.base.BaseEstimator
    splits : int
        number of folds for cross validation
    upsample : str or None
        Can specify either 'ADASYN' or 'SMOTE' or 'random'
        If 'random' is specified, then the features aren't used
        (you'll probably do this for text)

    Returns
    -------
    results : dict
        e.g.
        {'y_true': [np.array, ... ],
         'y_proba': [np.array, ... ],
         'models': [sklearn.model, ... ],
         'classes': ['directive', 'commissive' ... ]}
    """
    y_true = []
    y_proba = []
    models = []
    sss = StratifiedKFold(n_splits=splits, shuffle=True)
    for train_index, val_index in sss.split(X, y):
        print('Training')
        x_train, x_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model.fit(x_train, y_train)
        if probability:
            proba = model.predict_proba(x_val)
        else:
            proba = model.predict(x_val)
        y_true.append(y_val)
        y_proba.append(proba)
        models.append(model)
    classes = model.classes_
    return {
        'y_true': y_true,
        'y_proba': y_proba,
        'models': models,
        'classes': classes
    }
