from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from .transforms import IQRCapper
from .config import (
    NUMERIC_COLS,
    NOMINAL_COLS,
    ORDINAL_COLS,
    ORDINAL_CATEGORIES,
    OTHER_NOMINAL_COLS,
)

""" Generate preprocessor """
def generate_preprocessor(is_apply_scaler: bool = True) -> ColumnTransformer:
    # Pipeline numérique
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("outliers", IQRCapper())
    ]

    if is_apply_scaler:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_pipeline = Pipeline(numeric_steps)

    # Pipeline nominal
    nominal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Pipeline ordinal
    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OrdinalEncoder(
                categories=ORDINAL_CATEGORIES,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
        ),
    ])

    # column transformer
    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_COLS),
        ("nom", nominal_pipeline, NOMINAL_COLS + OTHER_NOMINAL_COLS),
        ("ord", ordinal_pipeline, ORDINAL_COLS),
    ])
