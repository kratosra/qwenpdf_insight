from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./test",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    num_train_epochs=1,
)
print("TrainingArguments loaded successfully")
