
import pandas as pd
from datasets import Dataset
from transformers import (
    XLMRobertaTokenizerFast,
    XLMRobertaForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
import numpy as np
import evaluate

# 1. Charger les données
df = pd.read_csv("prompt_classification_dataset_50000.csv")
df = df.dropna()
label_list = sorted(df["label"].unique().tolist())
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}
df["labels"] = df["label"].map(label2id)

# 2. Split train/test
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df["labels"])
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# 3. Tokenizer et model
model_name = "xlm-roberta-base"
tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_name)
model = XLMRobertaForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# 4. Tokenisation
def tokenize_fn(example):
    return tokenizer(example["prompt"], truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(tokenize_fn, batched=True)
test_dataset = test_dataset.map(tokenize_fn, batched=True)

# Supprimer les colonnes inutiles pour Trainer
train_dataset = train_dataset.remove_columns(["__index_level_0__", "prompt", "label"])
test_dataset = test_dataset.remove_columns(["__index_level_0__", "prompt", "label"])

# 5. Évaluation
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

# 6. Entraînement
training_args = TrainingArguments(
    output_dir="./results_classifier",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()
