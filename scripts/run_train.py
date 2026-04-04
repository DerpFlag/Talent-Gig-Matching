from pathlib import Path
import sys
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train import train_matcher


def main() -> None:
    paths = yaml.safe_load(Path("configs/paths.yaml").read_text(encoding="utf-8"))
    cfg = yaml.safe_load(Path("configs/model.yaml").read_text(encoding="utf-8"))

    result = train_matcher(
        labels_path=paths["labels_path"],
        jobs_path=paths["processed_jobs_path"],
        resumes_path=paths["processed_resumes_path"],
        model_name=cfg["embedding_model_name"],
        max_length=int(cfg["max_length"]),
        batch_size=int(cfg["batch_size"]),
        epochs=int(cfg["epochs"]),
        lr_encoder=float(cfg["learning_rate_encoder"]),
        lr_head=float(cfg["learning_rate_head"]),
        weight_decay=float(cfg["weight_decay"]),
        validation_size=float(cfg["validation_size"]),
        random_seed=int(cfg["random_seed"]),
        positive_class_weight=float(cfg["positive_class_weight"]),
        model_output_path=cfg["model_output_path"],
    )
    print(f"Training complete. Best val loss: {result['best_val_loss']:.6f}")
    print(f"Model saved to: {result['model_output_path']}")


if __name__ == "__main__":
    main()
