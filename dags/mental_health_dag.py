from pathlib import Path
import json
import time

import joblib
import pendulum

from airflow.sdk import dag, task
from src.config import PROJECT_DIR, MODEL_PATH, DATA_PATH


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
        from src.data_loader import load_data_frame, split_data

        # Loeader data frame
        df = load_data_frame(DATA_PATH)

        # spliter les donners
        X_train, X_test, y_train, y_test = split_data(df)

        # genereate name of folder based a nanosecondes
        run_name = str(time.time_ns())
        run_dir = PROJECT_DIR / "artifacts" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Sauvegarder les données divisées dans un fichier
        split_path = run_dir / "split.joblib"
        joblib.dump((X_train, X_test, y_train, y_test), split_path)

        print("Train :", X_train.shape)
        print("Test :", X_test.shape)

        # returner le path de save
        return str(split_path)

    # ÉTAPE 2 : Entraînement et optimisation
    @task
    def train_models(split_path):
        from src.optimization import optimize_models, choose_best_model
        from src.persistence import save_model

        # loada data de train
        X_train, _, y_train, _ = joblib.load(split_path)

        # optimize models avec grid search
        tuned = optimize_models(X_train, y_train, cv=5)

        # Choizer baset model
        model_name, best_model = choose_best_model(tuned)

        model_path = Path(split_path).parent / "best_model.joblib"
        save_model(best_model, model_path)

        print("Meilleur modèle :", model_name)

        return {
            "name": model_name,
            "model_path": str(model_path),
        }

    # ÉTAPE 3 : Évaluation (meilleur modele)
    @task
    def evaluate_model_task(split_path, model_info):
        from src.evaluation import evaluate_model
        from src.persistence import load_model

        # loada data de test
        _, X_test, _, y_test = joblib.load(split_path)

        # loead baset model
        model = load_model(model_info["model_path"])

        # évaluate model
        metrics = evaluate_model(
            model_info["name"],
            model,
            X_test,
            y_test
        )

        # Path of metrics
        report_path = Path(split_path).parent / "metrics.json"

        # save object of metrics in file json
        report_path.write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )

        print("Métriques :", metrics)

        # return evaluated model path
        return model_info["model_path"]

    # ÉTAPE 4 : Sauvegarde (pour enrigistrer le meilleur modele dans le target path de model global dasn le projet)
    @task
    def save_final_model(model_path):
        from src.persistence import load_model, save_model

        # load modle from path de run
        model = load_model(model_path)

        # SAve model
        save_model(model, MODEL_PATH)

        print("Modèle sauvegardé :", MODEL_PATH)

    # Définition de l'ordre des tâches
    split_file = prepare_data()
    best_info = train_models(split_file)
    evaluated_path = evaluate_model_task(split_file, best_info)
    save_final_model(evaluated_path)

# Run pipeline
mental_health_pipeline()