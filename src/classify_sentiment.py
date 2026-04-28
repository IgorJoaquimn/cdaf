import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 1. Configurações
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "./bertimbau-bundesliga"
input_path = 'data/processed/consolidated/bundesliga_2425_cazetv_chat.parquet'
output_path = 'data/processed/consolidated/bundesliga_chat_with_sentiment.parquet'

print(f"Carregando modelo de: {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()

# 2. Preparar Dataset para PyTorch (Performance)
class CommentDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return str(self.texts[idx])

df = pd.read_parquet(input_path)
print(f"Dataset carregado: {len(df):,} linhas.")

dataset = CommentDataset(df['mensagem'].tolist())
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

# 3. Inferência em Lote
predictions = []
print("Iniciando classificação automática...")

with torch.no_grad():
    for batch in tqdm(dataloader):
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        outputs = model(**inputs)
        batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        predictions.extend(batch_preds)

# 4. Salvar resultados
df['sentiment_pred'] = predictions
sentiment_map = {0: 'Negativo', 1: 'Neutro', 2: 'Positivo'}
df['sentiment_label'] = df['sentiment_pred'].map(sentiment_map)

# Converter para tipos padrão para evitar problemas com o motor parquet
df_export = df.copy()
for col in df_export.columns:
    df_export[col] = df_export[col].astype(object)

df_export.to_parquet(output_path, index=False, engine='fastparquet')

print(f"\nClassificação concluída!")
print(f"Dataset final salvo em: {output_path}")
print("\nDistribuição de sentimentos detectada:")
print(df['sentiment_label'].value_counts())
