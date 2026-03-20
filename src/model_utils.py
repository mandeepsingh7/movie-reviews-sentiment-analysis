import numpy as np 
import evaluate 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config import MODEL_NAME, MAX_LENGTH, NUM_LABELS

def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(batch, tokenizer):
    return tokenizer(
        batch['text'],
        truncation = True,
        padding = 'max_length',
        max_length = MAX_LENGTH
    )

def tokenize_dataset(dataset, tokenizer):
    return {
        split: dataset[split].map(
            lambda batch: tokenize_function(batch, tokenizer),
            batched=True,
            remove_columns = ['text']
        )
        for split in dataset
    }

def get_model():
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels = NUM_LABELS
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    metric = evaluate.load('accuracy')
    return metric.compute(predictions=preds, references=labels)