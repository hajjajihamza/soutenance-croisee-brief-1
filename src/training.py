from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from .transforms import SocialMediaCapper
from .config import RANDOM_STATE
from .preprocessing import generate_preprocessor

def build_model(model, is_apply_scaler: bool = True) -> Pipeline:
    return Pipeline([
        ("fix_social_media", SocialMediaCapper()),
        ("preprocessing", generate_preprocessor(is_apply_scaler)),
        ("classifier", model)
    ])


def get_models() -> dict:
    return {
        "Logistic Regression": build_model(
            LogisticRegression(max_iter=1000)
        ),

        "KNN": build_model(
            KNeighborsClassifier(n_neighbors=5)
        ),

        "Random Forest": build_model(
            RandomForestClassifier(
                random_state=RANDOM_STATE
            ),
            is_apply_scaler=False,
        ),
    }


def train_model(model, X_train, y_train):
    return model.fit(X_train, y_train)
