# Prédiction du risque en santé mentale chez les étudiants

> **Projet pédagogique de Data Science et de Machine Learning** — Classification supervisée, analyse exploratoire, pipelines, orchestration Airflow et visualisation Streamlit.

## Présentation

Ce projet vise à explorer un jeu de données portant sur des étudiants afin d'étudier les facteurs **associés** au bien-être et au risque en santé mentale, puis à construire et comparer des prototypes de classification supervisée.

Le travail couvre l'ensemble d'un processus de Data Science : audit des données, nettoyage, analyse exploratoire (EDA), préparation, entraînement, optimisation, évaluation, automatisation et visualisation.

> [!IMPORTANT]
> **Usage strictement pédagogique.** Les prédictions ne constituent ni un diagnostic, ni un dépistage clinique, ni une estimation médicale individuelle fiable. Le prototype ne doit pas servir à des décisions disciplinaires, à la sélection d'étudiants ou à l'attribution automatique d'un soutien. Les résultats sont susceptibles de contenir des erreurs et des biais.

## Jeu de données

- **Source :** [Mental Health Risk Prediction Dataset – Students (Kaggle)](https://www.kaggle.com/datasets/maqdadali/mental-health-risk-prediction-dataset-students)
- **Format et colonnes :** à vérifier après téléchargement et chargement du jeu de données.
- **Variable cible :** à identifier et documenter pendant l'audit initial.
- **Licence et conditions d'utilisation :** à consulter sur la page Kaggle avant toute redistribution.

Ne publiez pas d'informations personnelles ou de données sensibles dans le dépôt. Le dossier `data/` doit être géré selon les conditions du jeu de données et les règles de confidentialité applicables.

## Objectifs

1. **Explorer** les données et identifier les facteurs statistiquement associés au risque étudié.
2. **Préparer** les variables en évitant toute fuite de données (*data leakage*).
3. **Comparer** au moins trois modèles de classification sur une base méthodologique commune.
4. **Optimiser et évaluer** les modèles, puis conserver le pipeline retenu.
5. **Automatiser** le flux de traitement avec Apache Airflow et **présenter** les résultats dans une application Streamlit.

L'analyse d'associations statistiques ne démontre pas de relation de causalité.

## Technologies envisagées

| Domaine | Outils |
| --- | --- |
| Langage | Python 3 |
| Manipulation des données | pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine Learning | scikit-learn |
| Gestion du déséquilibre (si nécessaire) | imbalanced-learn / SMOTE |
| Sauvegarde du pipeline | joblib ou pickle |
| Orchestration | Apache Airflow |
| Application interactive | Streamlit |
| Suivi du travail | Jira, Git et GitHub |

## Étapes du projet

### 1. Chargement, audit et analyse exploratoire (EDA)

- Examiner les dimensions, types de colonnes et premières observations.
- Détecter les données manquantes, doublons, incohérences et valeurs aberrantes.
- Calculer les statistiques descriptives.
- Étudier la distribution de la cible et l'équilibre des classes.
- Visualiser les relations entre les variables, la cible et les corrélations pertinentes.

**Livrables attendus :** notebook EDA, visualisations et synthèse des observations.

### 2. Nettoyage, transformation et préparation

- Séparer **avant l'apprentissage des transformations** les données en **80 % entraînement / 20 % test** (stratification si appropriée).
- Mettre en place l'imputation des valeurs manquantes selon le type de variable.
- Examiner les valeurs aberrantes et justifier leur traitement.
- Encoder les variables catégorielles et standardiser ou normaliser les variables numériques si nécessaire.
- Évaluer le besoin de rééquilibrage (SMOTE, sur- ou sous-échantillonnage), **sur les seules données d'entraînement et dans les plis de validation croisée**.
- Regrouper les traitements dans un `Pipeline` / `ColumnTransformer` (ou `imblearn.pipeline.Pipeline` si échantillonnage).

**Livrables attendus :** pipeline de préparation reproductible et justification des choix.

### 3. Entraînement et comparaison des modèles

- Entraîner au moins **trois modèles** de classification (par exemple : régression logistique, arbre de décision et random forest).
- Comparer les modèles avec une stratégie de validation croisée commune sur l'entraînement.
- Mesurer l'**accuracy**, la **precision**, le **recall**, le **F1-score**, la **matrice de confusion** et, si approprié, le **ROC-AUC**.
- Adapter les métriques à la distribution des classes et aux conséquences possibles des erreurs.

**Livrables attendus :** tableau comparatif des modèles et analyse des métriques.

### 4. Optimisation et évaluation finale

- Retenir les modèles les plus prometteurs.
- Rechercher les hyperparamètres avec `GridSearchCV` ou `RandomizedSearchCV`.
- Choisir le pipeline final sur la base de la validation effectuée **sans utiliser le test pour sélectionner le modèle**.
- Évaluer une seule fois le pipeline retenu sur le jeu de test indépendant.
- Analyser les erreurs, les biais potentiels, les limites et les risques de mauvaise interprétation.
- Sauvegarder le pipeline complet dans `models/` (`.joblib` ou `.pkl`).

**Livrables attendus :** modèle sérialisé, rapport d'évaluation et matrice de confusion.

### 5. Airflow, Streamlit et livraison

- Définir un DAG Airflow pour orchestrer la préparation, l'entraînement et l'évaluation.
- Planifier son exécution et surveiller les statuts et les erreurs.
- Développer une interface Streamlit pour consulter les métriques, les graphiques et la matrice de confusion.
- Proposer une démonstration de prédiction sur de nouvelles observations **fictives ou autorisées**, accompagnée d'un avertissement sur les limites du modèle.
- Vérifier la reproductibilité et préparer la présentation devant le jury blanc.

**Livrables attendus :** DAG, interface de démonstration et documentation.

## Planification

**Période du projet : du 09/11/2026 au 14/11/2026 à 23 h 59.** Travail individuel ; évaluation : **jury blanc**.

| Date | Travail prioritaire |
| --- | --- |
| 09/11/2026 | Chargement, audit et EDA |
| 10/11/2026 | Nettoyage et pipeline de préparation |
| 11/11/2026 | Entraînement et comparaison de modèles |
| 12/11/2026 | Optimisation, évaluation et sauvegarde |
| 13/11/2026 | Orchestration Airflow et interface Streamlit |
| 14/11/2026 | Tests, documentation et préparation de la présentation |

Les tâches peuvent être suivies dans Jira avec des clés du type `SCB1-1`, `SCB1-2`, etc.

## Arborescence cible (proposée)

L'arborescence suivante est **une proposition d'organisation**, et non une liste de fichiers déjà implémentés :

```text
mental-health-risk-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
├── dags/
│   └── ml_pipeline_dag.py
├── models/
│   └── .gitkeep
├── reports/
│   └── figures/
├── app/
│   └── streamlit_app.py
└── tests/
```

## Installation et exécution (après implémentation)

Les commandes suivantes sont **indicatives**. Elles supposent que les fichiers mentionnés dans l'arborescence ont été créés.

```bash
# Créer un environnement virtuel
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

# Installer les dépendances (une fois requirements.txt créé)
pip install -r requirements.txt

# Lancer l'application (une fois développée)
streamlit run app/streamlit_app.py
```

L'installation et le démarrage d'Airflow dépendent de l'environnement retenu (local, conteneurs, etc.). La procédure exacte devra être ajoutée dès que sa configuration sera définie.

## Bonnes pratiques méthodologiques

- Fixer une graine aléatoire (`random_state`) pour faciliter la reproductibilité.
- Garder le jeu de test isolé jusqu'à l'évaluation finale.
- Ajuster l'imputation, l'encodage, la mise à l'échelle et le rééchantillonnage **uniquement sur l'entraînement** (et au sein des plis de validation croisée).
- Éviter les variables révélant directement la cible ou indisponibles au moment d'une prédiction.
- Étudier le déséquilibre des classes et ne pas se limiter à l'accuracy.
- Documenter les limites du jeu de données, la représentativité des échantillons et les éventuels biais.
- Ne charger que des fichiers `.pkl` ou `.joblib` provenant de sources fiables : la désérialisation peut exécuter du code.

## État du projet

**Statut : planification / démarrage.** Les modèles, résultats chiffrés et composants Airflow/Streamlit seront ajoutés au fil de l'implémentation. Aucune performance de prédiction n'est annoncée à ce stade.

## Référence

Source des données : [Kaggle — Mental Health Risk Prediction Dataset – Students](https://www.kaggle.com/datasets/maqdadali/mental-health-risk-prediction-dataset-students).
