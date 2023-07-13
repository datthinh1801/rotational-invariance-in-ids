import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


def data_prep_openml(X_train, y_train, seed: int = None):
    cat_idxs, cont_idxs = get_column_types(X_train)
    cat_dims = compute_cat_dims(X_train, cat_idxs)

    train_mean, train_std = compute_train_stats(X_train, cont_idxs)
    train_nan_mask = compute_nan_mask(X_train)

    X_train, y_train = data_split(X_train, y_train, train_nan_mask)

    return (
        cat_dims,
        cat_idxs,
        cont_idxs,
        X_train,
        y_train,
        train_mean,
        train_std,
    )


def get_column_types(X):
    categorical_columns = []
    cont_columns = []

    for col_idx in range(X.shape[1]):
        if np.issubdtype(X[:, col_idx].dtype, np.number):
            cont_columns.append(col_idx)
        else:
            categorical_columns.append(col_idx)

    return categorical_columns, cont_columns


def compute_train_stats(X_train, cont_idxs):
    train_mean = np.array(X_train[:, cont_idxs], dtype=np.float32).mean(axis=0)
    train_std = np.array(X_train[:, cont_idxs], dtype=np.float32).std(axis=0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)
    return train_mean, train_std


def compute_nan_mask(X):
    temp = pd.DataFrame(X).fillna("MissingValue")
    nan_mask = temp.ne("MissingValue").astype(int)
    return nan_mask


def compute_cat_dims(X_train, cat_idxs):
    cat_dims = [
        len(X_train[:, idx].astype("category").cat.categories) for idx in cat_idxs
    ]
    return cat_dims


def data_split(X, y, nan_mask):
    x_d = {"data": X, "mask": nan_mask.values}

    if x_d["data"].shape != x_d["mask"].shape:
        raise ValueError("Shape of data is not the same as that of nan mask!")

    y_d = {"data": y.reshape(-1, 1)}
    return x_d, y_d


class DataSetCatCon(Dataset):
    def __init__(self, X, Y, cat_cols, continuous_mean_std=None):
        X_mask = X["mask"].copy()
        X = X["data"].copy()

        cat_cols = cat_cols if cat_cols is not None else []
        con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))

        self.X1 = X[:, cat_cols].copy().astype(np.int32)  # categorical columns
        self.X2 = X[:, con_cols].copy().astype(np.float32)  # numerical columns
        self.X1_mask = (
            X_mask[:, cat_cols].copy().astype(np.int32)
        )  # categorical columns
        self.X2_mask = X_mask[:, con_cols].copy().astype(np.int32)  # numerical columns
        self.y = Y["data"]
        self.cls = np.zeros_like(self.y, dtype=int)
        self.cls_mask = np.ones_like(self.y, dtype=int)

        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            np.concatenate((self.cls[idx], self.X1[idx])),
            self.X2[idx],
            self.y[idx],
            np.concatenate((self.cls_mask[idx], self.X1_mask[idx])),
            self.X2_mask[idx],
        )
