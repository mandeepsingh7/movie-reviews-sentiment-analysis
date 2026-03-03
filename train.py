import argparse 
import json 
import os 
import numpy as np 
import evaluate 
from dotenv import load_dotenv

from datasets import load_dataset 
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

load_dotenv()

# ------------------
# Argument Parsing 
# ------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Movie Reviews Sentiment Analysis Training")

    parser.add_argument('--model_name', type=str, default="distilbert/distilbert-base-uncased")
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--output_dir', type=str, default="./results/distilbert_3")

    return parser.parse_args()

def main():
    args = parse_args()

    # Load Dataset 
    dataset = load_dataset('imdb')

    # Tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_function(batch):
        return tokenizer(
            batch['text'],
            truncation = True,
            max_length = 256
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

    # Model 
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    # Data Collator (Dynamic Padding instead of fixed)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Accuracy Metrics 
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    # Training Arguments

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay
    )

    # Trainer 
    trainer = Trainer(
        model=model, 
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

    history_path = os.path.join(args.output_dir, "train_history.json")
    with open(history_path, "w") as f:
        json.dump(trainer.state.log_history, f)

    # Evaluation
    eval_results = trainer.evaluate()
    print("Evaluation Results:")
    print(eval_results)
    eval_results_path = os.path.join(args.output_dir, "eval_results.json")
    with open(eval_results_path, "w") as f:
        json.dump(eval_results, f)

    # Save model 
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == '__main__':
    main()
