import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

from .config import TARGET


VALID_RANGES = {
    "Age": (0, 30),
    "Sleep_Hours": (0, 24),
    "Study_Hours_Per_Week": (0, 168),
    "Social_Media_Hours": (0, 24),
    "Screen_Time_Hours": (0, 24),
    "Attendance_Rate": (0, 100),
}

""" report sur les missing values """
def missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "count": df.isna().sum(),
        "percent": df.isna().mean() * 100,
    })


""" la somme des valeurs duplicer """
def duplicate_count(df: pd.DataFrame):
    return df.duplicated().sum()


""" report sur les invalid values base sur valide range de chque colonne """
def invalid_values_report(df: pd.DataFrame) -> DataFrame | None:
    invalids= []
    for col , (min, max) in VALID_RANGES.items():
        invalids.append({
            'column': col,
            'sum': (df[col].notna() & ~df[col].between(min,max)).sum()
        })

    return pd.DataFrame(invalids)


""" afficher cont des valeurs outlier de chque columns """
def iqr_outlier_report(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = []

    for col in columns:
        # calculer q1 et q3 et iqr
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1

        # calculer lower et upper
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        result.append({
            'column': col,
            'outliers': ((df[col] < lower) | (df[col] > upper)).sum(),
            'lower': lower,
            'upper': upper
        })
    return pd.DataFrame(result)


def figure_scatterplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str = TARGET,
) -> None:

    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
    )

    plt.tight_layout()
    plt.show()
