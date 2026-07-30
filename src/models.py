from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def get_logistic():
    return LogisticRegression(
        max_iter=100,
        class_weight="balanced",
        random_state=42,
    )

def get_random_forest():
    return RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=42,
    )

def get_xgboost():
    return XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.05,
        reg_lambda=10,
        # min_child_weight=2,
        random_state=42,
    )

MODELS = {
    "logistic": get_logistic,
    "rf": get_random_forest,
    "xgb": get_xgboost,
}