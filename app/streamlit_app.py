"""Application Streamlit du projet Mental Health Risk Prediction.

Lancer à la racine du projet : uv run streamlit run app.py
Dépend des fichiers src/ et des sorties produites par le DAG Airflow.
"""
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay

from src.config import (
    NUMERIC_COLS, NOMINAL_COLS, ORDINAL_COLS,
    ORDINAL_CATEGORIES, OTHER_NOMINAL_COLS,
)
from src.persistence import load_model


ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "mental_health_pipeline.joblib"
DATA_PATH = ROOT / "data" / "row.csv"
ARTIFACTS_DIR = ROOT / "artifacts"

st.set_page_config(page_title="Mental Health Risk", page_icon="🎓", layout="wide")
st.title("🎓 Mental Health Risk Prediction")
st.caption("Projet pédagogique de classification — aucun résultat ne constitue un diagnostic médical.")


@st.cache_resource
def get_model(path: str, modified: float):
    """Charger une fois le modèle; modifier son fichier invalide le cache."""
    return load_model(path)


@st.cache_data
def get_data(path: str, modified: float):
    return pd.read_csv(path)


def latest_report():
    """Trouver le dernier rapport Airflow disponible."""
    reports = list(ARTIFACTS_DIR.glob("*/metrics.json"))
    if not reports:
        return None
    return max(reports, key=lambda p: p.stat().st_mtime)


def categorical_choices(column, df):
    """Valeurs trouvées dans le CSV, sinon valeurs par défaut."""
    if column in ORDINAL_COLS:
        return ORDINAL_CATEGORIES[ORDINAL_COLS.index(column)]
    if df is not None and column in df.columns:
        vals = df[column].dropna().unique().tolist()
        if vals:
            return sorted(vals, key=str)
    defaults = {
        "Gender": ["Male", "Female"],
        "Part_Time_Job": ["Yes", "No"],
        "Previous_Mental_Health_History": ["Yes", "No"],
        "Year_of_Study": ["1st Year", "2nd Year", "3rd Year", "4th Year"],
    }
    return defaults.get(column, ["Unknown"])


def numeric_input(column, df):
    """Contrôle numérique avec valeur par défaut issue du CSV si présent."""
    defaults = {
        "Age": (18.0, 15.0, 100.0),
        "Sleep_Hours": (7.0, 0.0, 24.0),
        "Study_Hours_Per_Week": (20.0, 0.0, 168.0),
        "GPA": (3.0, 0.0, 4.0),
        "Social_Media_Hours": (2.0, 0.0, 24.0),
        "Screen_Time_Hours": (5.0, 0.0, 24.0),
        "Financial_Stress": (3.0, 0.0, 10.0),
        "Family_Support": (3.0, 0.0, 10.0),
        "Attendance_Rate": (85.0, 0.0, 100.0),
    }
    value, low, high = defaults[column]
    if df is not None and column in df.columns:
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if not values.empty:
            low = min(low, float(values.min()))
            high = max(high, float(values.max()))
            value = float(values.median())
    return st.number_input(column, min_value=float(low), max_value=float(high),
                           value=float(max(low, min(value, high))), step=0.5)


# Charger les ressources facultatives
sample_df = None
if DATA_PATH.exists():
    try:
        sample_df = get_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
    except Exception as exc:
        st.warning(f"Impossible de lire data/row.csv : {exc}")

if not MODEL_PATH.exists():
    st.warning("Aucun modèle trouvé. Exécute d'abord le DAG Airflow pour créer models/mental_health_pipeline.joblib.")
    st.stop()

try:
    model = get_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"Impossible de charger le modèle : {exc}")
    st.stop()

st.sidebar.success("Modèle chargé")
st.sidebar.caption(str(MODEL_PATH.relative_to(ROOT)))
st.sidebar.info("Le modèle et les rapports doivent provenir d'une source fiable.")

metrics_tab, predict_tab = st.tabs(["📊 Résultats du modèle", "🔮 Tester une prédiction"])

with metrics_tab:
    st.subheader("Résultats du dernier entraînement Airflow")
    report_file = latest_report()
    if report_file is None:
        st.info("Pas encore de rapport metrics.json. Exécute le DAG Airflow.")
    else:
        try:
            report = json.loads(report_file.read_text(encoding="utf-8"))
            st.caption(f"Rapport : {report_file.relative_to(ROOT)}")
            cols = st.columns(4)
            for widget, key in zip(cols, ["Accuracy", "Precision", "Recall", "F1-score"]):
                value = report.get(key)
                if value is not None:
                    widget.metric(key, f"{float(value):.1%}")
            st.dataframe(pd.DataFrame([report]), use_container_width=True, hide_index=True)
        except (OSError, ValueError, TypeError) as exc:
            st.warning(f"Rapport non lisible : {exc}")

        # La matrice utilise le candidat et le split du même run.
        run_dir = report_file.parent
        split_file = run_dir / "split.joblib"
        candidate_file = run_dir / "candidate.joblib"
        if split_file.exists() and candidate_file.exists():
            st.subheader("Matrice de confusion du run")
            try:
                _, X_test, _, y_test = joblib.load(split_file)
                candidate = joblib.load(candidate_file)
                y_pred = candidate.predict(X_test)
                fig, ax = plt.subplots(figsize=(7, 5))
                ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
                st.pyplot(fig)
                plt.close(fig)
            except Exception as exc:
                st.warning(f"Matrice indisponible : {exc}")
        else:
            st.info("Pour la matrice de confusion, conserver split.joblib et candidate.joblib du même run.")

with predict_tab:
    st.subheader("Saisir les données d'un étudiant")
    st.write("Les catégories doivent correspondre à celles utilisées à l'entraînement.")
    with st.form("prediction_form"):
        features = {}
        left, right = st.columns(2)
        for i, col in enumerate(NUMERIC_COLS):
            with (left if i % 2 == 0 else right):
                features[col] = numeric_input(col, sample_df)
        st.markdown("**Variables catégorielles**")
        categorical_cols = NOMINAL_COLS + OTHER_NOMINAL_COLS + ORDINAL_COLS
        for i, col in enumerate(categorical_cols):
            with (left if i % 2 == 0 else right):
                choices = categorical_choices(col, sample_df)
                features[col] = st.selectbox(col, choices)
        submitted = st.form_submit_button("Prédire", type="primary")

    if submitted:
        if features["Social_Media_Hours"] > features["Screen_Time_Hours"]:
            st.warning("Les heures de réseaux sociaux dépassent le temps d'écran. Le pipeline appliquera sa correction.")
        input_df = pd.DataFrame([features])
        try:
            prediction = model.predict(input_df)[0]
            st.success(f"Classe prédite : {prediction}")
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                labels = model.classes_
                probability_df = pd.DataFrame({
                    "Classe": labels.astype(str),
                    "Probabilité estimée": probabilities,
                }).set_index("Classe")
                st.bar_chart(probability_df)
                st.caption("Ces probabilités sont des estimations statistiques, pas des probabilités cliniques validées.")
            with st.expander("Voir les données utilisées"):
                st.dataframe(input_df, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Échec de la prédiction : {exc}")

st.divider()
st.caption("Utilisation pédagogique uniquement — ne pas employer pour décider d'une intervention concernant un étudiant.")