from sklearn import metrics as met
import pandas as pd

def evaluate_model(y_true, y_pred, y_prob, verbose=True):
    """Evaluate a binary classification model."""

    metrics = {
        "accuracy": met.accuracy_score(y_true, y_pred),
        "precision": met.precision_score(y_true, y_pred),
        "recall": met.recall_score(y_true, y_pred),
        "f1": met.f1_score(y_true, y_pred),
        "roc_auc": met.roc_auc_score(y_true, y_prob),
        "pr_auc": met.average_precision_score(y_true, y_prob),
    }

    if verbose:
        print(f"Accuracy  = {metrics['accuracy']:.4f}")
        print(f"Precision = {metrics['precision']:.4f}")
        print(f"Recall    = {metrics['recall']:.4f}")
        print(f"F1        = {metrics['f1']:.4f}")
        print(f"ROC-AUC   = {metrics['roc_auc']:.4f}")
        print(f"PR-AUC    = {metrics['pr_auc']:.4f}")
        
        print("---------------")
        
        print("Confusion Matrix:")
        print(met.confusion_matrix(y_true, y_pred))

    return metrics


def find_best_threshold(model, X, y_true, thresholds):
    """Find the threshold that maximizes F1 score."""

    y_prob = model.predict_proba(X)[:, 1]
    results = []
    best_threshold = thresholds[0]
    best_f1 = -1

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        metrics = evaluate_model(y_true, y_pred, y_prob, verbose=False)

        results.append({
            "Threshold": threshold,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"],
        })

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = threshold

    results = pd.DataFrame(results)
    print(results)
    print(f"\nBest Threshold: {best_threshold}")
    return best_threshold