import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    classification_report,
    ConfusionMatrixDisplay,
)

def evaluate_model(name: str, model, X_test, y_test, plot: bool = True) -> dict:
    # prediction
    y_pred = model.predict(X_test)

    # calculate les motrics
    result = {
        "model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred,
            average="macro",
            zero_division=0,
        ),
        "Recall": recall_score(
            y_test, y_pred,
            average="macro",
            zero_division=0,
        ),
        "F1-score": f1_score(
            y_test, y_pred,
            average="macro",
            zero_division=0,
        ),
    }

    print(f"\n=== {name} ===")

    # Classification report
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0,
        )
    )

    if plot:
        # Classification multiclasse : One-vs-Rest
        probabilities = model.predict_proba(X_test) # probabilitie
        classes = model.classes_

        plt.figure(figsize=(8, 6))

        # calculate auc and dsiplay graf de ROC Curve de chaque class
        for i, cls in enumerate(classes):
            y_binary = (np.array(y_test) == cls).astype(int) # Converter les boolean to nombers 0, 1

            # calculer False Positive Rate et True Positive Rate et Thresholds (_)
            fpr, tpr, _ = roc_curve(y_binary, probabilities[:, i])

            # Caculate AUC (entre 0 et 1)
            class_auc = auc(fpr, tpr)

            plt.plot(
                fpr, tpr,
                label=f"{cls} : AUC = {class_auc:.3f}"
            )

        # Diagonale
        plt.plot([0, 1], [0, 1], "--", color="gray")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {name}")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Calculate auc score final
        auc_score = roc_auc_score(
            y_test,
            probabilities,
            multi_class="ovr", # One vs Rest
            average="macro",
            labels=classes
        )

        # Ajouter dans metrics
        result['ROC_AUC'] = auc_score


        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            labels=model.classes_,
            cmap="Blues",
        )

        plt.title(f"Matrice de confusion - {name}")
        plt.tight_layout()
        plt.show()

    return result


def error_analysis(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mistakes = X_test.copy()
    mistakes["Actual"] = y_test
    mistakes["Predicted"] = y_pred

    return mistakes[
        mistakes["Actual"] != mistakes["Predicted"]
    ]
