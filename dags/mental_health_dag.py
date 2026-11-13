from pathlib import Path
import hashlib
import json

import joblib
import pendulum

from airflow.sdk import dag, task, get_current_context


# Chemins du projet
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "row.csv"


@dag(
    dag_id="mental_health_pipeline",
    start_date=pendulum.datetime(2026, 11, 1, tz="UTC"),
    schedule=None,
    catchup=False,
    tags=["machine-learning", "students"],
)
def mental_health_pipeline():

    # ÉTAPE 1 : Chargement et préparation
    @task
    def prepare_data():
        from src.config import TARGET
        from src.data_loader import load_data_frame, split_data

        df = load_data_frame(DATA_PATH)

        if df.empty or TARGET not in df.columns or df[TARGET].isna().any():
            raise ValueError("Dataset vide ou cible absente/incomplète")

        X_train, X_test, y_train, y_test = split_data(df)

        if y_train.value_counts().min() < 5:
            raise ValueError("Pas assez d'exemples pour CV=5")

        run_id = get_current_context()["run_id"]
        run_name = hashlib.sha256(run_id.encode()).hexdigest()[:12]
        run_dir = PROJECT_DIR / "artifacts" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        split_path = run_dir / "split.joblib"
        joblib.dump((X_train, X_test, y_train, y_test), split_path)

        print("Train :", X_train.shape)
        print("Test :", X_test.shape)

        return str(split_path)

    # ÉTAPE 2 : Entraînement et optimisation
    @task
    def train_models(split_path):
        from src.optimization import optimize_models, choose_best_model
        from src.persistence import save_model

        X_train, _, y_train, _ = joblib.load(split_path)

        results = optimize_models(X_train, y_train, cv=5)

        model_name, best_model = choose_best_model(results)

        model_path = Path(split_path).parent / "candidate.joblib"
        save_model(best_model, model_path)

        print("Meilleur modèle :", model_name)

        return {
            "name": model_name,
            "model_path": str(model_path),
        }

    # ÉTAPE 3 : Évaluation
    @task
    def evaluate_model_task(split_path, model_info):
        from src.evaluation import evaluate_model
        from src.persistence import load_model

        _, X_test, _, y_test = joblib.load(split_path)

        model = load_model(model_info["model_path"])

        metrics = evaluate_model(
            model_info["name"],
            model,
            X_test,
            y_test,
            plot=False,
        )

        report_path = Path(split_path).parent / "metrics.json"

        report_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("Métriques :", metrics)

        return model_info["model_path"]

    # ÉTAPE 4 : Sauvegarde
    @task
    def save_final_model(model_path):
        from src.persistence import load_model, save_model

        model = load_model(model_path)

        target_path = (
            PROJECT_DIR / "models" / "mental_health_pipeline.joblib"
        )

        save_model(model, target_path)

        print("Modèle sauvegardé :", target_path)

    # Définition de l'ordre des tâches
    split_file = prepare_data()
    best = train_models(split_file)
    evaluated_path = evaluate_model_task(split_file, best)
    save_final_model(evaluated_path)


mental_health_pipeline()