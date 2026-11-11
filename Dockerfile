
FROM apache/airflow:2.9.2-python3.12

WORKDIR /opt/airflow

# Copier la configuration du projet
COPY --chown=airflow:0 pyproject.toml ./pyproject.toml

# Installer les dépendances avec uv
RUN uv pip install --no-cache \
    --python /home/airflow/.local/bin/python \
    "apache-airflow==2.9.2" \
    -r pyproject.toml

# Vérifier l'installation
RUN python -c "import pandas, sklearn, joblib; print('Dependencies OK')"
