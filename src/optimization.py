from sklearn.model_selection import GridSearchCV
from .training import get_models


PARAM_GRIDS = {
    "Logistic Regression": {
        "classifier__C": [0.1, 1, 10],
        "classifier__class_weight": [None, "balanced"],
    },

    "KNN": {
        "classifier__n_neighbors": [3, 5, 7, 9, 11],
        "classifier__weights": ["uniform", "distance"],
        "classifier__metric": ["euclidean", "manhattan"],
    },

    "Random Forest": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [5, 10, None],
    },
}


def generate_optimizer(pipeline, param_grid, cv: int = 5):
    return GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )


def optimize_models(X_train, y_train, cv: int = 5):
    tuned = {}

    for name, model in get_models().items():

        grid = generate_optimizer(
            model,
            PARAM_GRIDS[name],
            cv=cv,
        )

        grid.fit(X_train, y_train)

        tuned[name] = grid

        print(
            f"{name}: "
            f"best F1 CV={grid.best_score_:.3f}; "
            f"params={grid.best_params_}"
        )

    return tuned


def choose_best_model(tuned: dict):
    name = max(
        tuned,
        key=lambda key: tuned[key].best_score_, # tuned[key] is grid
    )

    return name, tuned[name].best_estimator_
