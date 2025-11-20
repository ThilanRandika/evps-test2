# Conflict Minimizer Model Card

## Intended Use
Classify upcoming intersection states as **Safe** or **Unsafe** for emergency vehicle preemption.

## Data
- Source: Phase 1 conflict snapshots (`data/raw/conflicts/*.csv`).
- Engineered features: count of conflicting lanes, queue statistics, speed statistics, moving lane fraction, current signal phase. All scaled to [0,1] with MinMaxScaler.
- Labels: Rule-based heuristic combining queue congestion and moving-lane activity (see `docs/PHASE2_LABELING_RULES.md`).
- Split: Stratified 70/15/15 train/val/test.

## Model
- Pipeline: MinMaxScaler + DecisionTreeClassifier.
- Hyperparameters tuned via GridSearchCV over `max_depth` [3,5,7,9,None] and `min_samples_leaf` [1,5,10].
- Best model saved to `models/conflict_classifier.pkl`.

## Evaluation
- Metrics: Accuracy, Precision, Recall on test set; confusion matrix stored at `artifacts/metrics/conflict_confusion_matrix.png`.

## Limitations
- Labels are heuristic approximations; may not capture complex intersection dynamics.
- Decision trees may overfit if input distribution shifts; consider pruning or ensembles for production use.
