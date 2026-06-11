import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    # Set styles
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.titlesize': 18
    })
    
    # 1. Load human golden set
    df_golden = pd.read_csv('data/processed/consolidated/golden_set_consensus.csv')
    df_golden['mensagem_clean'] = df_golden['mensagem'].str.strip()
    
    # 2. Load Llama predictions
    df_llama = pd.read_csv('data/processed/consolidated/llama_eval_results.csv')
    df_llama['mensagem_clean'] = df_llama['mensagem'].str.strip()
    
    # Merge for Llama eval (handling many-to-many duplicates correctly)
    eval_df = df_golden.merge(df_llama[['mensagem_clean', 'pred']], on='mensagem_clean').drop_duplicates(subset=['mensagem_clean']).copy()
    
    y_true = eval_df['label'].values
    y_pred_llama = eval_df['pred'].values
    
    # 3. Load BERTimbau model & predict
    model_path = 'bertimbau-bundesliga'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    
    messages = eval_df['mensagem'].tolist()
    y_pred_bert = []
    batch_size = 64
    for i in range(0, len(messages), batch_size):
        batch = messages[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
            y_pred_bert.extend(batch_preds)
            
    eval_df['pred_bertimbau'] = y_pred_bert
    y_pred_bert = np.array(y_pred_bert)
    
    # Ensure reports directory exists
    os.makedirs('reports', exist_ok=True)
    
    # --- Plot 1: Confusion Matrix Llama 3.1 8B FP8 ---
    plt.figure(figsize=(7, 6))
    cm_llama = confusion_matrix(y_true, y_pred_llama)
    sns.heatmap(cm_llama, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negativo', 'Neutro', 'Positivo'],
                yticklabels=['Negativo', 'Neutro', 'Positivo'])
    plt.title('Matriz de Confusão: Llama 3.1 8B FP8')
    plt.ylabel('Classe Real (Humana)')
    plt.xlabel('Classe Predita (Llama)')
    plt.tight_layout()
    plt.savefig('reports/confusion_matrix_llama.png', dpi=300)
    plt.close()
    
    # --- Plot 2: Confusion Matrix BERTimbau ---
    plt.figure(figsize=(7, 6))
    cm_bert = confusion_matrix(y_true, y_pred_bert)
    sns.heatmap(cm_bert, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Negativo', 'Neutro', 'Positivo'],
                yticklabels=['Negativo', 'Neutro', 'Positivo'])
    plt.title('Matriz de Confusão: BERTimbau (Destilado)')
    plt.ylabel('Classe Real (Humana)')
    plt.xlabel('Classe Predita (BERTimbau)')
    plt.tight_layout()
    plt.savefig('reports/confusion_matrix_bertimbau.png', dpi=300)
    plt.close()
    
    # --- Plot 3: Distribution of Sentiments (Comparison) ---
    plt.figure(figsize=(10, 6))
    
    # Read the 2k synthetic dataset for BERT distribution
    df_synthetic = pd.read_csv('data/processed/consolidated/synthetic_dataset_for_bert.csv')
    
    # Prepare counts
    golden_counts = df_golden['label'].value_counts().sort_index()
    synthetic_counts = df_synthetic['label'].value_counts().sort_index()
    
    # Normalize to percentages
    golden_pct = (golden_counts / golden_counts.sum()) * 100
    synthetic_pct = (synthetic_counts / synthetic_counts.sum()) * 100
    
    categories = ['Negativo', 'Neutro', 'Positivo']
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, golden_pct, width, label='Golden Set (Humano, N=652)', color='#3498db')
    rects2 = ax.bar(x + width/2, synthetic_pct, width, label='Treino Sintético (Llama, N=2000)', color='#2ecc71')
    
    ax.set_ylabel('Porcentagem (%)')
    ax.set_title('Distribuição de Sentimentos nos Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    # Attach labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
            
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig('reports/sentiment_distribution.png', dpi=300)
    plt.close()
    
    print("Plots generated successfully!")
    print("- reports/confusion_matrix_llama.png")
    print("- reports/confusion_matrix_bertimbau.png")
    print("- reports/sentiment_distribution.png")

if __name__ == "__main__":
    main()
