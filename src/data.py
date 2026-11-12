"""Data loading and preprocessing for the GBSG2 survival analysis."""

import numpy as np
import pandas as pd
from lifelines.datasets import load_gbsg2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
FEATURES = ["horTh", "age", "menostat", "tsize", "pnodes",
            "progrec", "estrec", "tgrade_II", "tgrade_III"]
NUM_COLS = ["age", "tsize", "pnodes", "progrec", "estrec"]
LOG_COLS = ["pnodes", "progrec", "estrec"]


def prepare_data(seed=SEED, test_size=0.2):
    """Load GBSG2 and return a stratified, standardised train/test split.

    Returns X_train, X_test, y_train, y_test (with columns time and event)
    and the fitted StandardScaler.
    """
    df = load_gbsg2().rename(columns={"cens": "event"})

    df["horTh"] = (df["horTh"] == "yes").astype(int)
    df["menostat"] = (df["menostat"] == "Post").astype(int)
    df = pd.get_dummies(df, columns=["tgrade"], drop_first=True, dtype=int)

    for col in LOG_COLS:
        df[col] = np.log1p(df[col])

    X = df[FEATURES]
    y = df[["time", "event"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y["event"], random_state=seed
    )

    X_train, X_test = X_train.copy(), X_test.copy()
    scaler = StandardScaler()
    X_train[NUM_COLS] = scaler.fit_transform(X_train[NUM_COLS])
    X_test[NUM_COLS] = scaler.transform(X_test[NUM_COLS])

    return X_train, X_test, y_train, y_test, scaler