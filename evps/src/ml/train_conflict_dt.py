import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from .common import METRICS_DIR, MODELS_DIR, PROCESSED_DIR, mae, rmse, save_json, set_seed

FEATURE_COLS = [
    "num_conflicting_lanes",
    "max_queue",
    "mean_queue",
    "sum_queue",
    "std_queue",
    "max_mean_speed",
    "mean_mean_speed",
    "moving_lane_fraction",
    "current_signal_phase",
]


def load_split(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"conflict_{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed split {path}")
    return pd.read_csv(path)


def train_model(seed: int):
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["label"]

    pipeline = Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("clf", DecisionTreeClassifier(random_state=seed)),
        ]
    )
    param_grid = {
        "clf__max_depth": [3, 5, 7, 9, None],
        "clf__min_samples_leaf": [1, 5, 10],
    }
    grid = GridSearchCV(pipeline, param_grid=param_grid, cv=5, scoring="accuracy")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    joblib.dump(best_model, MODELS_DIR / "conflict_classifier.pkl")

    X_test = test_df[FEATURE_COLS]
    y_test = test_df["label"]
    preds = best_model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, pos_label="Unsafe"),
        "recall": recall_score(y_test, preds, pos_label="Unsafe"),
        "n_test_samples": int(len(y_test)),
    }
    metrics_path = METRICS_DIR / "conflict_metrics.json"
    save_json(metrics_path, metrics)

    disp = ConfusionMatrixDisplay.from_predictions(y_test, preds, display_labels=["Safe", "Unsafe"])
    plt.title("Conflict Classifier Confusion Matrix")
    cm_path = METRICS_DIR / "conflict_confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    print(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {metrics_path}")
    print(f"Model saved to {MODELS_DIR / 'conflict_classifier.pkl'}")

    # optional tree export
    try:
        tree: DecisionTreeClassifier = best_model.named_steps["clf"]
        dot_path = Path("docs/conflict_tree.dot")
        png_path = Path("docs/conflict_tree.png")
        export_graphviz(
            tree,
            out_file=dot_path,
            feature_names=FEATURE_COLS,
            class_names=best_model.classes_,
            filled=True,
            rounded=True,
        )
        try:
            import pydot

            (graph,) = pydot.graph_from_dot_file(str(dot_path))
            graph.write_png(str(png_path))
        except Exception:
            pass
    except Exception:
        pass

    return metrics


def main(seed: int = 42):
    set_seed(seed)
    train_model(seed=seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train conflict decision tree")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(seed=args.seed)
