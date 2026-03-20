import argparse 
import sys 
from pathlib import Path 
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.load_data import load_imdb_dataset
from src.model_utils import get_tokenizer, tokenize_dataset, get_model, compute_metrics
from src.train_utils import build_trainer, train_and_evaluate
from src.config import TRANSFORMERS_MODELS_DIR

def parse_args():
    parser = argparse.ArgumentParser(description='Train a sentiment analysis model on the IMDB dataset.')

    parser.add_argument('--run_name', type=str, required=True, help='Name of the training run.')
    parser.add_argument('--data_fraction', type=float, default=1.0, help='Fraction of the training dataset to use for training and validation.')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')

    return parser.parse_args()

def main():
    args = parse_args()

    dataset = load_imdb_dataset(args.data_fraction)

    tokenizer = get_tokenizer()
    encoded_dataset = tokenize_dataset(dataset, tokenizer)

    model = get_model()

    output_dir = TRANSFORMERS_MODELS_DIR / args.run_name

    trainer = build_trainer(
        model, 
        tokenizer,
        encoded_dataset['train'],
        encoded_dataset['val'],
        compute_metrics,
        output_dir,
        args.epochs, 
        args.lr,
        args.weight_decay
    )

    train_and_evaluate(trainer)

if __name__ == "__main__":
    main()

