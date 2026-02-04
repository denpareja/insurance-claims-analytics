import joblib
import pandas as pd

from config import get_paths


def main() -> None:
    paths = get_paths()
    if not paths.model_path.exists():
        raise FileNotFoundError(
            "No model found. Run: python src/train_model.py")

    model = joblib.load(paths.model_path)

    # Example input (you can change these values)
    example = pd.DataFrame([{
        "age": 40,
        "sex": "female",
        "bmi": 27.5,
        "children": 1,
        "smoker": "no",
        "region": "southeast"
    }])

    pred = float(model.predict(example)[0])

    print("✅ Prediction for example input:")
    print(example)
    print(f"\n✅ Predicted charges: {pred:.2f}")

    # Save a small file with the example + prediction
    out = example.copy()
    out["predicted_charges"] = pred
    out_path = paths.outputs_dir / "prediction_one_example.csv"
    out.to_csv(out_path, index=False)

    print(f"\n✅ Saved: {out_path}")


if __name__ == "__main__":
    main()
