from pathlib import Path
import joblib, shap, json
from datetime import datetime
from src.evaluation import evaluate_model, find_best_threshold
from src.models import MODELS
from src.preprocessing import build_pipline, load_data
from sklearn.metrics import roc_curve, precision_recall_curve

MODEL_PATH = Path("models/xgb.pkl")
THRESHOLDS = [0.5, 0.7, 0.9]

def train():
    # load data
    X_train, X_test, X_dev, y_train, y_test, y_dev = load_data("data/raw/paysim.csv")

    # train
    pipeline = build_pipline(MODELS["xgb"]())
    pipeline.fit(X_train, y_train)
    best_t = find_best_threshold(pipeline, X_dev, y_dev, THRESHOLDS)

    # evaluate on dev
    y_prob_dev = pipeline.predict_proba(X_dev)[:, 1]
    y_pred_dev = (y_prob_dev >= best_t).astype(int)

    # evaluate on test
    y_prob_test = pipeline.predict_proba(X_test)[:, 1]
    y_pred_test = (y_prob_test >= best_t).astype(int)

    # some samples for test
    samples = X_test.copy()
    samples["fraud_probability"] = y_prob_test
    samples["isFraud"] = y_test.values

    def nearest(df, target):
        idx = (df["fraud_probability"] - target).abs().idxmin()
        return df.loc[idx]

    sample_specs = [
        ("🟢 Safe Payment", 0, 0.01),
        ("🟢 Normal Transfer", 0, 0.10),
        ("🟠 Borderline", None, 0.50),
        ("🔴 Likely Fraud", 1, 0.85),
        ("🔴 Extreme Fraud", 1, 0.99),
    ]

    demo = {}

    for name, label, target in sample_specs:
        if label is None:
            pool = samples
        else:
            pool = samples[samples["isFraud"] == label]

        row = nearest(pool, target)
        demo[name] = {
            "type": row["type"],
            "amount": float(row["amount"]),
            "oldbalanceOrg": float(row["oldbalanceOrg"]),
            "newbalanceOrig": float(row["newbalanceOrig"]),
            "oldbalanceDest": float(row["oldbalanceDest"]),
            "newbalanceDest": float(row["newbalanceDest"]),
            "step": int(row["step"]),
            "expected_probability": float(row["fraud_probability"]),
            "true_label": int(row["isFraud"]),
        }

    with open("models/demo_samples.json", "w") as f:
        json.dump(demo, f, indent=4)

    # eval
    print("\nCross Validation:\n")
    metrics = evaluate_model(y_dev, y_pred_dev, y_prob_dev, verbose=True)

    print("\nTest:\n")
    evaluate_model(y_test, y_pred_test, y_prob_test, verbose=True)

    # save model performance
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # save the model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    # save the explainer
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    joblib.dump(explainer, "models/explainer.pkl")

    # save metadata
    config = {
        "threshold": float(best_t),
        "model": "xgb",
        "version": "0.1.0",
        "trained_date": datetime.now().strftime("%Y-%m-%d"),
        "features": X_train.shape[1]
    }

    with open("models/config.json", "w") as f:
        json.dump(config, f, indent=4)

    fpr, tpr, _ = roc_curve(y_test, y_prob_test)
    precision, recall, _ = precision_recall_curve(y_test, y_prob_test)

    with open("models/curves.json", "w") as f:
        json.dump(
            {
                "roc": {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                },
                "pr": {
                    "precision": precision.tolist(),
                    "recall": recall.tolist(),
                },
            },
            f,
        )

    print("\nModel and config saved")

if __name__ == "__main__":
    train()
