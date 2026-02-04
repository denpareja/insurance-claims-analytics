from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path
    data_path: Path
    outputs_dir: Path
    model_path: Path
    metrics_path: Path
    preds_path: Path
    feat_imp_path: Path


def get_paths() -> Paths:
    root = Path(__file__).resolve().parents[1]  # project root
    data_path = root / "data" / "insurance.csv"
    outputs_dir = root / "outputs"
    return Paths(
        root=root,
        data_path=data_path,
        outputs_dir=outputs_dir,
        model_path=outputs_dir / "model.joblib",
        metrics_path=outputs_dir / "metrics.json",
        preds_path=outputs_dir / "predictions_sample.csv",
        feat_imp_path=outputs_dir / "feature_importance.csv",
    )
