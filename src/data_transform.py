import ipdb
import pandas as pd
import numpy as np

from scipy.stats import special_ortho_group
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def preprocess_data(
    X: pd.DataFrame,
    y: pd.Series,
    class_list: list,
    drop_cols: list = None,
    seed=None,
):
    """Preprocess data and return the training/validation/testing sets."""
    # drop pre-defined columns
    if drop_cols is not None:
        X.drop(columns=drop_cols, inplace=True)

    # replace np.inf values
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # impute missing values
    mean_imputer_X = SimpleImputer(strategy="mean")
    X = mean_imputer_X.fit_transform(X)
    # X = pd.DataFrame(X_imputed, columns=X.columns)

    mode_imputer_y = SimpleImputer(strategy="most_frequent")
    y = mode_imputer_y.fit_transform(
        y.values.reshape(-1, 1)
    )  # as SimpleImputer imputes values by each column, y must be reshape to (-1, 1)
    # y = pd.Series(y_imputed.flatten(), name=y.name)

    # scaling the data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # label-encode the target column
    mapping = mapping = {
        class_name: index for index, class_name in enumerate(class_list)
    }
    y = np.vectorize(mapping.get)(y)
    y = np.reshape(y, -1)

    return X, y


def feature_selection(
    X: np.ndarray,
    y: np.array,
    seed=None,
):
    """Perform feature selection using Random Forest."""
    # create a Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=seed)
    rf.fit(X, y)

    # Perform feature selection
    selector = SelectFromModel(rf, threshold="mean")
    selector.fit(X, y)

    # Get the selected features
    # selected_features = selector.get_support(indices=True)

    # Transform the training and testing data to include only the selected features
    X = selector.transform(X)
    return X, y


def apply_random_rotation(X_train, X_test, seed=None):
    """Apply random rotation on x_train and x_val."""
    num_features = X_train.shape[1]
    rotation_matrix = special_ortho_group.rvs(num_features, random_state=seed)

    X_train = X_train @ rotation_matrix
    X_test = X_test @ rotation_matrix

    return (X_train, X_test)
