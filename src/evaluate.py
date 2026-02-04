import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from config import get_paths
from utils import TARGET_COL, load_data, split_xy, evaluate_regression


def main() -> None:
    paths = get_paths()
    if not paths.model_path.exists():
        raise FileNotFoundError(
            "No model found. Run: python src/train_model.py")

    df = load_data(paths.data_path)
    X, y = split_xy(df, TARGET_COL)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = joblib.load(paths.model_path)
    preds = model.predict(X_test)
    metrics = evaluate_regression(y_test, preds)

    print("✅ Evaluation metrics on test set:")
    print(metrics)


if __name__ == "__main__":
    main()
