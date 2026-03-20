from pathlib import Path 
import json 

from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from src.config import TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE 

def build_trainer(
        model, 
        tokenizer,
        train_dataset,
        eval_dataset,
        compute_metrics,
        output_dir,
        epochs,
        lr,
        weight_decay    
):
    training_args = TrainingArguments(
        output_dir = str(output_dir),
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        learning_rate=lr,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model = model, 
        args = training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    return trainer

def train_and_evaluate(trainer):
    trainer.train()
    
    output_dir = Path(trainer.args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(output_dir))

    with open(output_dir / "training_log.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=4)

    results = trainer.evaluate()

    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print('Evaluation results:', results)
    return results 
