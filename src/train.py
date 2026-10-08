"""Train, tune, evaluate and track heart disease models with MLflow."""
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline

from src.preprocessing import (CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET,
                               build_preprocessor)
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "heart_clean.csv"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "screenshots" / "training"
RANDOM_STATE = 42

MODELS = {
    "logistic_regression": (
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        {"model__C": [0.01, 0.1, 1, 10]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {"model__n_estimators": [100, 200],
         "model__max_depth": [3, 5, None],
         "model__min_samples_split": [2, 5]},
    ),
}


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "roc_auc"]

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("heart-disease-classification")

    best_name, best_auc, best_pipeline = None, -1, None

    for name, (estimator, grid) in MODELS.items():
        with mlflow.start_run(run_name=name):
            pipe = Pipeline([("prep", build_preprocessor()), ("model", estimator)])
            search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
            search.fit(X_train, y_train)
            best = search.best_estimator_

            cv_res = cross_validate(best, X_train, y_train, cv=cv, scoring=scoring)
            pred = best.predict(X_test)
            proba = best.predict_proba(X_test)[:, 1]

            mlflow.log_param("model_type", name)
            mlflow.log_params({k: str(v) for k, v in search.best_params_.items()})
            for m in scoring:
                mlflow.log_metric(f"cv_{m}_mean", cv_res[f"test_{m}"].mean())
                mlflow.log_metric(f"cv_{m}_std", cv_res[f"test_{m}"].std())
            test_auc = roc_auc_score(y_test, proba)
            mlflow.log_metric("test_accuracy", accuracy_score(y_test, pred))
            mlflow.log_metric("test_precision", precision_score(y_test, pred))
            mlflow.log_metric("test_recall", recall_score(y_test, pred))
            mlflow.log_metric("test_f1", f1_score(y_test, pred))
            mlflow.log_metric("test_roc_auc", test_auc)

            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax)
            ax.set_title(f"{name} - confusion matrix")
            cm_path = FIG_DIR / f"{name}_confusion_matrix.png"
            fig.savefig(cm_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots()
            RocCurveDisplay.from_predictions(y_test, proba, ax=ax)
            ax.set_title(f"{name} - ROC curve")
            roc_path = FIG_DIR / f"{name}_roc_curve.png"
            fig.savefig(roc_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(roc_path))
            mlflow.sklearn.log_model(
                best,
                name="model",
                serialization_format="cloudpickle",
            )

            print(f"{name}: CV AUC={cv_res['test_roc_auc'].mean():.3f} "
                  f"| test AUC={test_auc:.3f} | params={search.best_params_}")

            if test_auc > best_auc:
                best_name, best_auc, best_pipeline = name, test_auc, best

    joblib.dump(best_pipeline, MODEL_DIR / "model.joblib")
    print(f"Best model: {best_name} (test AUC {best_auc:.3f}) -> models/model.joblib")


if __name__ == "__main__":
    main()
