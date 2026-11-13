from pathlib import Path
import joblib


def save_model(
    model,
    path="models/mental_health_pipeline.joblib",
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(model, path)

    return path


def load_model(
    path="models/mental_health_pipeline.joblib",
):
    return joblib.load(path)
