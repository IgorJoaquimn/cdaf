import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import numpy as np
import evaluate

# 1. Configurações
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "neuralmind/bert-base-portuguese-cased"
output_dir = "./bertimbau-bundesliga"

# 2. Carregar e preparar dados
df = pd.read_csv('data/processed/consolidated/gold_standard_labeled.csv')
dataset = Dataset.from_pandas(df[['mensagem', 'sentiment_manual']].rename(columns={'sentiment_manual': 'label'}))
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# 3. Tokenização
tokenizer = AutoTokenizer.from_pretrained(model_name)
def tokenize_function(examples):
    return tokenizer(examples["mensagem"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 4. Modelo
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)

# 5. Métricas
f1_metric = evaluate.load("f1")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return f1_metric.compute(predictions=predictions, references=labels, average="macro")

# 6. Treinamento
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=10, # Mais épocas para um dataset pequeno
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

print("Iniciando Fine-tuning...")
trainer.train()

# 7. Salvar modelo final
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Modelo treinado e salvo em: {output_dir}")
