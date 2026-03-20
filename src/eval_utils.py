import torch 
import tqdm 
import json 
import matplotlib.pyplot as plt 
from transformers import AutoTokenizer, AutoModelForSequenceClassification 


def load_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer

def predict(texts, model, tokenizer, batch_size=32):
    all_preds = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating"):
        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,   # important
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)

        all_preds.extend(preds.cpu().numpy())

    return all_preds

def plot_training_curves(log_path, save_path):
    import json
    import matplotlib.pyplot as plt

    if not log_path.exists():
        print("No training_log.json found.")
        return

    with open(log_path) as f:
        logs = json.load(f)

    train_loss, val_loss, val_acc = [], [], []
    epochs_train, epochs_val = [], []

    for log in logs:
        if "loss" in log and "epoch" in log:
            train_loss.append(log["loss"])
            epochs_train.append(log["epoch"])

        if "eval_loss" in log and "epoch" in log:
            val_loss.append(log["eval_loss"])
            val_acc.append(log["eval_accuracy"])
            epochs_val.append(log["epoch"])

    plt.figure(figsize=(10, 4))

    # ---- Loss Plot ----
    plt.subplot(1, 2, 1)
    plt.plot(epochs_train, train_loss, label="Train")
    plt.plot(epochs_val, val_loss, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)

    # ---- Accuracy Plot ----
    plt.subplot(1, 2, 2)
    plt.plot(epochs_val, val_acc, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()