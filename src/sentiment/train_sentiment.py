import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import numpy as np
import evaluate
import os

def main():
    # 1. Configurações
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "neuralmind/bert-base-portuguese-cased"
    output_dir = "./bertimbau-bundesliga"
    synthetic_data_path = 'data/processed/consolidated/synthetic_dataset_for_bert.csv'
    golden_data_path = 'data/processed/consolidated/golden_set_consensus.csv'

    # 2. Carregar e preparar dados
    print("Carregando datasets...")
    df_train = pd.read_csv(synthetic_data_path)
    df_eval = pd.read_csv(golden_data_path)

    # Dataset de treino (Sintético - 5000 exemplos)
    train_dataset = Dataset.from_pandas(df_train[['mensagem', 'label']])
    
    # Dataset de validação (Golden Set Humano - ~650 exemplos)
    eval_dataset = Dataset.from_pandas(df_eval[['mensagem', 'label']])

    # 3. Tokenização
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize_function(examples):
        return tokenizer(examples["mensagem"], padding="max_length", truncation=True, max_length=128)

    print("Tokenizando...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

    # 4. Modelo
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)

    # 5. Métricas
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        # Weighted F1 é melhor para desbalanceamento
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
        acc = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
        return {"f1": f1, "accuracy": acc}

    # 6. Treinamento
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5, 
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
    )

    print("Iniciando Fine-tuning do BERTimbau...")
    trainer.train()

    # 7. Salvar e Avaliar Final
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    results = trainer.evaluate()
    print("\n--- Resultados Finais no Golden Set (Humano) ---")
    print(f"Accuracy: {results['eval_accuracy']:.4f}")
    print(f"F1-Score (Weighted): {results['eval_f1']:.4f}")
    print(f"Modelo salvo em: {output_dir}")

if __name__ == "__main__":
    main()
