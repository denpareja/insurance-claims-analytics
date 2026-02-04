import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from config import get_paths
from utils import (
    TARGET_COL, ensure_dir, load_data, split_xy,
    get_feature_types, build_pipeline, evaluate_regression,
    save_json, try_feature_importance
)


def train() -> None:
    paths = get_paths()
    ensure_dir(paths.outputs_dir)

    print(f"✅ Loading data: {paths.data_path}")
    df = load_data(paths.data_path)

    X, y = split_xy(df, TARGET_COL)
    numeric_cols, categorical_cols = get_feature_types(X)

    print(f"✅ Numeric cols: {numeric_cols}")
    print(f"✅ Categorical cols: {categorical_cols}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = build_pipeline(numeric_cols, categorical_cols)

    print("✅ Training model...")
    pipe.fit(X_train, y_train)

    print("✅ Evaluating...")
    preds = pipe.predict(X_test)
    metrics = evaluate_regression(y_test, preds)
    metrics["n_train"] = int(len(X_train))
    metrics["n_test"] = int(len(X_test))

    # Save model
    joblib.dump(pipe, paths.model_path)

    # Save metrics
    save_json(metrics, paths.metrics_path)

    # Save sample predictions (first 50 rows of X_test)
    sample = X_test.head(50).copy()
    sample["predicted_charges"] = pipe.predict(sample)
    sample.to_csv(paths.preds_path, index=False)

    # Try feature importance
    try:
        preprocess = pipe.named_steps["preprocess"]
        feature_names = preprocess.get_feature_names_out()
        df_imp = try_feature_importance(pipe, list(feature_names))
        if not df_imp.empty:
            df_imp.to_csv(paths.feat_imp_path, index=False)
            print(f"✅ Feature importance saved to: {paths.feat_imp_path}")
    except Exception as e:
        print(f"⚠️ Could not save feature importance: {e}")

    print("✅ Training complete.")
    print("✅ Model saved to:", paths.model_path)
    print("✅ Metrics saved to:", paths.metrics_path)
    print("✅ Sample predictions saved to:", paths.preds_path)
    print("✅ Metrics:", metrics)


if __name__ == "__main__":
    train()
