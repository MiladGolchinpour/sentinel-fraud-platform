from pathlib import Path
import joblib, shap, json
from datetime import datetime
from evaluation import evaluate_model, find_best_threshold
from models import MODELS
from preprocessing import build_pipline, load_data

MODEL_PATH = Path("models/xgb.pkl")
THRESHOLDS = [0.5, 0.7, 0.9]

def main():
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

    print("\nModel and config saved")

if __name__ == "__main__":
    main()
