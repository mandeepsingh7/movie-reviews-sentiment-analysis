import argparse 
import json 
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.eval_utils import load_model, predict, plot_training_curves
from src.config import TEST_DATA_PATH 

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a dataset")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the trained model directory")
    return parser.parse_args()

def main():
    args = parse_args()

    model_dir = Path(args.model_dir)
    save_dir = model_dir / "evaluation"
    save_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(model_dir)

    df = pd.read_csv(TEST_DATA_PATH)

    preds = predict(df["text"].tolist(), model, tokenizer)

    accuracy = (preds == df["label"].values).mean()

    results = {
        "test_accuracy": float(accuracy)
    }

    with open(save_dir / "test_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Test Accuracy:", accuracy)

    plot_training_curves(
        model_dir / "training_log.json",
        save_dir / "training_curves.png"
    )

if __name__ == "__main__":
    main()