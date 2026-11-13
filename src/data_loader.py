from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import TARGET, ID_COL, RANDOM_STATE


""" loader data frame """
def load_data_frame(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)

""" spliter features and target """
def split_features_and_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET, ID_COL], errors="ignore")
    y = df[TARGET]

    return X, y

""" spliter data frame """
def split_data(df: pd.DataFrame, test_size: float = 0.2):
    X, y = split_features_and_target(df)

    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )
