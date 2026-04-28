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
# Usando precisão mista (half precision) para velocidade e economia de VRAM
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).half()
model.eval()

# 2. Preparar Dataset otimizado
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
# Batch size aumentado para 512 para ocupar os 16GB de VRAM
dataloader = DataLoader(dataset, batch_size=512, shuffle=False, pin_memory=True)

# 3. Inferência de Alta Performance
predictions = []
print("Iniciando classificação automática (Modo Turbo - FP16)...")

with torch.no_grad():
    for batch in tqdm(dataloader):
        # Tokenização em lote
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        # Inferência
        with torch.autocast(device_type='cuda'):
            outputs = model(**inputs)
            batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            predictions.extend(batch_preds)

# 4. Salvar resultados
df['sentiment_pred'] = predictions
sentiment_map = {0: 'Negativo', 1: 'Neutro', 2: 'Positivo'}
df['sentiment_label'] = df['sentiment_pred'].map(sentiment_map)

# Exportação segura
df_export = df.copy()
for col in df_export.columns:
    df_export[col] = df_export[col].astype(object)

df_export.to_parquet(output_path, index=False, engine='fastparquet')

print(f"\nClassificação concluída!")
print(f"Dataset final salvo em: {output_path}")
print("\nDistribuição de sentimentos:")
print(df['sentiment_label'].value_counts())
